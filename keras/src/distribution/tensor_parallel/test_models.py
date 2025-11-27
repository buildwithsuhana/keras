import logging
import os
import sys
import time
import numpy as np
import kagglehub # pip install kagglehub

# --- Project Root Setup ---
try:
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(_script_dir)))
    )
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
except Exception:
    pass

# --- Backend and Device Configuration ---
os.environ["KERAS_BACKEND"] = "jax"
# Force 8 devices logic if simulating on CPU
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import jax
import keras_hub
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import keras

# IMPORT AUTO-TP (Ensure this file exists in your path)
from keras.src.distribution.distribution_lib import AutoTPDistribution
from keras.src.distribution import DeviceMesh

tf.config.set_visible_devices([], "GPU")

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- JAX Device Detection ---
try:
    devices = jax.devices()
    logger.info(f"JAX devices found: {[str(d) for d in devices]}")
    TARGET_WORLD_SIZE = 2 # Split model across 2 logical devices
    TARGET_DEVICES = devices[:TARGET_WORLD_SIZE]
except Exception as e:
    logger.error(f"Could not initialize JAX. Error: {e}")
    TARGET_WORLD_SIZE = 0

# --- Constants ---
# Minimal settings to ensure 9B fits on Mac CPU
BATCH_SIZE = 1
SEQUENCE_LENGTH = 16 
LEARNING_RATE = 1e-4
EPOCHS = 1
STEPS_PER_EPOCH = 2
VALIDATION_STEPS = 2

MODEL_MAPPING = {
    "gemma2_9b_en": keras_hub.models.GemmaCausalLM, 
}

# ----------------------------------------------------------------------
# --- Dataset Helpers ---
# ----------------------------------------------------------------------

def load_shakespeare_dataset(model_preset):
    """Loads and preprocesses the Tiny Shakespeare dataset."""
    logger.info(f"Loading dataset for {model_preset}...")
    ds = tfds.load("tiny_shakespeare", split="train", as_supervised=False)
    text = "".join(example["text"].decode("utf-8") for example in ds.as_numpy_iterator())

    # Load tokenizer explicitly since we disable the model preprocessor
    tokenizer = keras_hub.models.GemmaTokenizer.from_preset(model_preset)
    
    token_ids = tokenizer.tokenize(text)
    num_tokens = (len(token_ids) // (SEQUENCE_LENGTH + 1)) * (SEQUENCE_LENGTH + 1)
    sequences = np.array(token_ids[:num_tokens]).reshape(-1, SEQUENCE_LENGTH + 1)

    all_data = tf.data.Dataset.from_tensor_slices(sequences)
    num_train_samples = int(0.9 * sequences.shape[0])

    train_ds = all_data.take(num_train_samples)
    val_ds = all_data.skip(num_train_samples)
    return train_ds, val_ds

def format_for_causal_lm(data):
    features = {
        "token_ids": data[:, :-1],
        "padding_mask": tf.ones_like(data[:, :-1], dtype=tf.bool),
    }
    labels = data[:, 1:]
    return features, labels

# ----------------------------------------------------------------------
# --- Main Verification Function ---
# ----------------------------------------------------------------------

def run_model_verification(preset_name, model_class):
    if TARGET_WORLD_SIZE < 2:
        return "SKIPPED"

    logger.info(f"--- VERIFICATION FOR: {preset_name.upper()} ---")

    # 1. Setup Distribution Strategy
    mesh = DeviceMesh(
        shape=(1, TARGET_WORLD_SIZE), 
        axis_names=("batch", "model"), 
        devices=TARGET_DEVICES
    )
    
    dist = AutoTPDistribution(
        device_mesh=mesh, 
        auto_shard_dataset=True
    )
    
    # 2. Lazy Initialization Scope
    logger.info("⏳ Starting Lazy Initialization Scope...")
    
    with dist.scope():
        # A. Create the model architecture (Empty Weights)
        # load_weights=False prevents OOM during init
        # preprocessor=None prevents double-tokenization errors
        model = model_class.from_preset(
            preset_name, 
            preprocessor=None, 
            load_weights=False 
        )
        
        # B. Trigger variable creation
        model.build((BATCH_SIZE, SEQUENCE_LENGTH))
        
        # C. ENABLE LORA (CRITICAL FOR MAC CPU)
        # This reduces trainable params from 9B to ~3MB
        # Without this, gradient allocation will OOM your Mac.
        logger.info("🔧 Enabling LoRA (Rank=4) to save memory...")
        model.backbone.enable_lora(rank=4)
        
        logger.info(f"✅ Model skeleton built. Trainable params: {model.count_params():,}")
        
        # D. Compile
        # Use SGD to avoid 2x memory overhead of Adam optimizer states
        model.compile(
            optimizer=keras.optimizers.SGD(learning_rate=0.01),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[
                keras_hub.metrics.Perplexity(from_logits=True, name="perplexity")
            ],
        )

    # 3. Download & Load Weights (Robust Sharding Support)
    logger.info(f"📥 Downloading weights for {preset_name}...")
    
    # Handle map for Kaggle models
    handle_map = {
        "gemma2_9b_en": "keras/gemma2/keras/gemma2_9b_en",
    }
    handle = handle_map.get(preset_name, f"keras/gemma2/keras/{preset_name}")
        
    try:
        model_dir = kagglehub.model_download(handle)
        
        # Check for Sharded Index vs Monolithic H5
        sharded_index = os.path.join(model_dir, "model.weights.json")
        monolithic_h5 = os.path.join(model_dir, "model.weights.h5")
        
        if os.path.exists(sharded_index):
            logger.info(f"found sharded weights index: {sharded_index}")
            model.load_weights(sharded_index)
        elif os.path.exists(monolithic_h5):
            logger.info(f"Found monolithic weights: {monolithic_h5}")
            model.load_weights(monolithic_h5)
        else:
             # Fallback for some KerasHub versions
            logger.warning("Attempting load via backbone...")
            model.backbone.load_weights(os.path.join(model_dir, "model.weights.h5"))

        logger.info("✅ Weights loaded successfully.")
        
    except Exception as e:
        logger.error(f"Failed to load weights: {e}")
        logger.warning("⚠️ Proceeding with RANDOM weights (Test mode).")

    # 4. Prepare Data
    train_ds_raw, val_ds_raw = load_shakespeare_dataset(preset_name)

    train_ds = (
        train_ds_raw.batch(BATCH_SIZE, drop_remainder=True)
        .map(format_for_causal_lm, num_parallel_calls=tf.data.AUTOTUNE)
    )
    val_ds = (
        val_ds_raw.batch(BATCH_SIZE, drop_remainder=True)
        .map(format_for_causal_lm, num_parallel_calls=tf.data.AUTOTUNE)
    )

    # 5. Distribute Dataset 
    train_ds = dist.distribute_dataset(train_ds)
    val_ds = dist.distribute_dataset(val_ds)

    # 6. Training Loop
    logger.info("\n--- Starting Training Loop (LoRA) ---")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_steps=VALIDATION_STEPS,
        verbose=1,
    )
    
    logger.info("✅ Training finished without OOM.")
    return True

# ----------------------------------------------------------------------
# --- Execution ---
# ----------------------------------------------------------------------

if __name__ == "__main__":
    keras.config.set_dtype_policy("mixed_bfloat16")
    
    print("\n" + "="*60)
    print(" GEMMA 2 9B - TENSOR PARALLEL + LORA VERIFICATION")
    print("="*60 + "\n")

    for preset, model_class in MODEL_MAPPING.items():
        try:
            run_model_verification(preset, model_class)
        except Exception as e:
            logger.error(f"CRITICAL FAIL: {e}", exc_info=True)