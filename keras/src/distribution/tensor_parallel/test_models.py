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
# Ensure we see all devices (Force host platform count for JAX simulation if needed)
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

import jax
import keras_hub
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import keras

# --- 1. TPU OPTIMIZATION: Bfloat16 ---
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
        # Fallback to GPU/CPU
        TARGET_DEVICES = devices
        logger.warning(f"⚠️ No TPUs found! Using {devices[0].platform}.")

    TARGET_WORLD_SIZE = len(TARGET_DEVICES)

except Exception as e:
    logger.error(f"Could not initialize JAX. Error: {e}")
    sys.exit(1)

from keras.src.distribution.distribution_lib import DeviceMesh
from keras.src.distribution.distribution_lib import AutoTPDistribution

# --- Constants ---
BATCH_SIZE = 1
SEQUENCE_LENGTH = 8
LEARNING_RATE = 5e-4
EPOCHS = 1
STEPS_PER_EPOCH = 10
VALIDATION_STEPS = 5

MODEL_MAPPING = {
    "opt_6.7b_en": keras_hub.models.OPTCausalLM,
}

# ----------------------------------------------------------------------
# --- Dataset Helpers ---
# ----------------------------------------------------------------------

def load_shakespeare_dataset(model_preset):
    """Loads and preprocesses the Tiny Shakespeare dataset."""
    logger.info(f"Loading dataset for {model_preset}...")
    ds = tfds.load("tiny_shakespeare", split="train", as_supervised=False)
    text = "".join(example["text"].decode("utf-8") for example in ds.as_numpy_iterator())

    # Tokenize on CPU
    with keras.device("cpu"):
        tokenizer = keras_hub.models.OPTTokenizer.from_preset(model_preset)
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
    def model_builder(**kwargs):
        logger.info(f"Creating {preset_name} model (inside scope)...")
        model = model_class.from_preset(preset_name, preprocessor=None, **kwargs)
        
        logger.info("--- Enabling LoRA (Rank=8) ---") 
        if hasattr(model, "enable_lora"):
             model.enable_lora(rank=8)
        elif hasattr(model, "backbone") and hasattr(model.backbone, "enable_lora"):
             model.backbone.enable_lora(rank=8)
        
        total_params = model.count_params()
        trainable_params = sum(np.prod(w.shape) for w in model.trainable_variables)
        
        logger.info(f"Model created. Total: {total_params:,} | Trainable: {trainable_params:,}")
        return model
    return model_builder

# ----------------------------------------------------------------------
# --- NEW: Physical Sharding Verification ("Truth Check") ---
# ----------------------------------------------------------------------
def inspect_physical_memory(model):
    """
    Directly checks JAX buffers on the device to prove sharding works.
    Calculates the Ratio of (Logical Global Size) / (Physical Local Size).
    """
    logger.info("\n" + "="*50)
    logger.info("🔍 PERFORMING PHYSICAL MEMORY INSPECTION")
    logger.info("="*50)
    
    target_var = None
    # Try to find a specific large weight (Value projection in Attention is usually a good target)
    for v in model.trainable_variables:
        # Check for path keywords to identify a kernel in self-attention
        if "self_attention" in v.path and "kernel" in v.path and "value" in v.path:
            target_var = v
            break
            
    # Fallback if specific layer not found
    if target_var is None:
        target_var = model.trainable_variables[0]

    # Get the underlying JAX array
    jax_array = target_var.value
    
    try:
        # jax_array.addressable_shards returns a list of shards stored on the *current* process's devices.
        # We look at the first one [0] to see how big the chunk is on a single chip.
        local_shard = jax_array.addressable_shards[0].data
        
        global_shape = jax_array.shape
        local_shape = local_shard.shape
        
        global_params = np.prod(global_shape)
        local_params = np.prod(local_shape)
        
        logger.info(f"   Target Variable:        {target_var.path}")
        logger.info(f"   Logical Shape (Global): {global_shape} ({global_params:,} params)")
        logger.info(f"   Physical Shape (Chip):  {local_shape}  ({local_params:,} params)")
        
        if local_params < global_params:
            ratio = global_params / local_params
            logger.info(f"   🎉 SUCCESS: Variable is physically sharded!")
            logger.info(f"   ⬇️  Memory footprint reduced by {ratio:.1f}x per chip.")
        else:
            logger.warning("   ⚠️ WARNING: Variable appears REPLICATED (Local size == Global size).")
            logger.warning("   Check if your layout heuristics matched this variable name.")
            
    except Exception as e:
        logger.error(f"   Could not inspect addressable shards. Error: {e}")
    
    logger.info("="*50 + "\n")


# ----------------------------------------------------------------------
# --- Main Verification Function ---
# ----------------------------------------------------------------------

def run_model_verification(preset_name, model_class):
    if TARGET_WORLD_SIZE < 2:
        logger.warning("Need >1 device for TP. Skipping.")
        # return "SKIPPED" # Uncomment if strictly skipping, else let it run for debug

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
    device_mesh = DeviceMesh(
        shape=(1, TARGET_WORLD_SIZE), 
        axis_names=('data', 'model'), 
        devices=TARGET_DEVICES
    )

    # 2. Prepare Builder
    model_builder_fn = model_builder_factory(preset_name, model_class)

    # 3. Initialize Strategy
    distribution = AutoTPDistribution(
        model_builder_fn, 
        device_mesh=device_mesh,
    )
    
    logger.info("\n--- Initializing & Sharding on TPU ---")

    # 4. Build & Distribute
    # Note: distribution.model automatically handles the scope internally, 
    # but the explicit scope here is safe and clear.
    with distribution.scope():
        tp_model = distribution.model 

    # ---------------------------------------------------------
    # 5. VERIFY SHARDING (The Truth Check)
    # ---------------------------------------------------------
    inspect_physical_memory(tp_model)
    # ---------------------------------------------------------

    # 6. Compile & Train
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
    
    logger.info("✅ SUCCESS: TPU Training Complete")
    return True

if __name__ == "__main__":
    # Run Verification
    try:
        run_model_verification("opt_6.7b_en", MODEL_MAPPING["opt_6.7b_en"])
    except Exception as e:
        logger.exception("TPU Run Failed")