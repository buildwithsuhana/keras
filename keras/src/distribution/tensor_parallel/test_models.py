import logging
import os
import sys
import time

import numpy as np

# --- Project Root Setup ---
try:
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(_script_dir)))
    )
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
except Exception:
    print("Could not add project root to sys.path.")

# --- Backend Configuration ---
os.environ["KERAS_BACKEND"] = "jax"

# TPU specific cleanup (Just in case)
# We DO NOT need XLA_PYTHON_CLIENT_MEM_FRACTION on TPUs.
# We DO want to ensure we see all 8 cores.

import jax
import keras_hub
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import keras

# --- 1. TPU OPTIMIZATION: Bfloat16 ---
# TPUs are natively optimized for bfloat16. 
# It provides the same range as float32 but with less memory.
keras.config.set_dtype_policy("mixed_bfloat16")

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- JAX Device Detection (TPU Logic) ---
try:
    devices = jax.devices()
    logger.info(f"JAX devices found: {[str(d) for d in devices]}")
    
    # Detect TPUs
    tpu_devices = [d for d in devices if d.platform == "tpu"]
    
    if tpu_devices:
        TARGET_DEVICES = tpu_devices
        logger.info(f"✅ Using {len(TARGET_DEVICES)} TPU Cores.")
    else:
        # Fallback to GPU/CPU (should not happen on v5e-8 node)
        TARGET_DEVICES = devices
        logger.warning(f"⚠️ No TPUs found! Using {devices[0].platform}.")

    TARGET_WORLD_SIZE = len(TARGET_DEVICES)

except Exception as e:
    logger.error(f"Could not initialize JAX. Error: {e}")
    sys.exit(1)

from keras.src.distribution import DeviceMesh
from keras.src.distribution.distribution_lib import AutoTPDistribution

# --- Constants (Scaled up for TPU) ---
# TPU v5e-8 has 128GB total memory. We can be aggressive.
BATCH_SIZE = 32        # 32 global batch size (4 per chip)
SEQUENCE_LENGTH = 128  # Back to full length
LEARNING_RATE = 1e-4
EPOCHS = 1
STEPS_PER_EPOCH = 10
VALIDATION_STEPS = 5

MODEL_MAPPING = {
    "gemma_7b_en": keras_hub.models.GemmaCausalLM,
}

# ----------------------------------------------------------------------
# --- Dataset Helpers ---
# ----------------------------------------------------------------------

def load_shakespeare_dataset(model_preset):
    """Loads and preprocesses the Tiny Shakespeare dataset."""
    logger.info(f"Loading dataset for {model_preset}...")
    ds = tfds.load("tiny_shakespeare", split="train", as_supervised=False)
    text = "".join(example["text"].decode("utf-8") for example in ds.as_numpy_iterator())

    # Tokenize on CPU to avoid TPU compilation overhead for simple ops
    with keras.device("cpu"):
        tokenizer = keras_hub.models.GemmaTokenizer.from_preset(model_preset)
        token_ids = tokenizer.tokenize(text)

    num_tokens = (len(token_ids) // (SEQUENCE_LENGTH + 1)) * (SEQUENCE_LENGTH + 1)
    sequences = np.array(token_ids[:num_tokens]).reshape(-1, SEQUENCE_LENGTH + 1)
    all_data = tf.data.Dataset.from_tensor_slices(sequences)
    
    # Split
    num_train = int(0.9 * sequences.shape[0])
    train_ds = all_data.take(num_train)
    val_ds = all_data.skip(num_train)
    return train_ds, val_ds

def format_for_causal_lm(data):
    features = {
        "token_ids": data[:, :-1],
        "padding_mask": tf.ones_like(data[:, :-1], dtype=tf.bool),
    }
    labels = data[:, 1:]
    return features, labels

# --- TPU MODEL BUILDER ---
def model_builder_factory(preset_name, model_class):
    """Builds the model. No Quantization needed for TPU v5e-8."""
    
    def model_builder(**kwargs):
        logger.info(f"Creating {preset_name} model (inside scope)...")
        
        # --- TPU CONFIGURATION ---
        # 1. No 'dtype' override needed (defaults to float32, policy casts to bfloat16).
        #    If you really want to save memory, use "bfloat16".
        #    Do NOT use "int8" on TPUs unless you specifically need to test quantization.
        
        # Create Model
        model = model_class.from_preset(preset_name, preprocessor=None, **kwargs)
        
        # 2. LoRA is OPTIONAL on TPU v5e-8 because we have 128GB RAM.
        #    However, keeping it makes training much faster.
        logger.info("--- Enabling LoRA (Rank=8) ---") 
        
        if hasattr(model, "enable_lora"):
             model.enable_lora(rank=8) # Increased rank for better quality
        elif hasattr(model, "backbone") and hasattr(model.backbone, "enable_lora"):
             model.backbone.enable_lora(rank=8)
        
        total_params = model.count_params()
        trainable_params = sum(np.prod(w.shape) for w in model.trainable_variables)
        
        logger.info(f"Model created. Total: {total_params:,} | Trainable: {trainable_params:,}")
        return model
        
    return model_builder

# ----------------------------------------------------------------------
# --- Main Verification Function ---
# ----------------------------------------------------------------------

def run_model_verification(preset_name, model_class):
    if TARGET_WORLD_SIZE < 2:
        logger.warning("Need >1 device for TP. Skipping.")
        return "SKIPPED"

    logger.info(f"--- RUNNING: {preset_name} on {TARGET_WORLD_SIZE} TPU CORES ---")

    # Data Pipeline
    train_ds_raw, val_ds_raw = load_shakespeare_dataset(preset_name)
    train_ds = (
        train_ds_raw.batch(BATCH_SIZE, drop_remainder=True)
        .map(format_for_causal_lm, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
        .repeat()
    )
    val_ds = (
        val_ds_raw.batch(BATCH_SIZE, drop_remainder=True)
        .map(format_for_causal_lm, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
        .repeat()
    )

    # 1. Initialize AutoTPDistribution
    # TPUs use a Mesh: (1, 8) for 8-way tensor parallelism
    device_mesh = DeviceMesh(
        shape=(1, TARGET_WORLD_SIZE), 
        axis_names=('data', 'model'), 
        devices=TARGET_DEVICES
    )

    # 2. Prepare Builder
    model_builder_fn = model_builder_factory(preset_name, model_class)

    # 3. Initialize Strategy
    # Ensure we pass float32 or bfloat16, NOT int8
    distribution = AutoTPDistribution(
        model_builder_fn, 
        device_mesh=device_mesh,
        # dtype="bfloat16" # Optional: Force weights to bf16 storage
    )
    
    logger.info("\n--- Initializing & Sharding on TPU ---")

    # 4. Build & Distribute
    with distribution.scope():
        tp_model = distribution.model 

    # 5. Compile & Train
    logger.info("🚀 Compiling...")
    tp_model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=LEARNING_RATE),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras_hub.metrics.Perplexity(from_logits=True, name="perplexity")
        ],
    )

    logger.info("🏃 Starting training loop...")
    tp_history = tp_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_steps=VALIDATION_STEPS,
        verbose=1,
    )
    
    # Plotting logic (same as before, omitted for brevity)
    # plot_training_graphs(tp_history, preset_name)
    
    logger.info("✅ SUCCESS: TPU Training Complete")
    return True

if __name__ == "__main__":
    # Run Verification
    try:
        run_model_verification("gemma_7b_en", MODEL_MAPPING["gemma_7b_en"])
    except Exception as e:
        logger.exception("TPU Run Failed")