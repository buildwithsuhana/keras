import logging
import os
import sys
import time
import gc
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
# Prevent JAX from pre-allocating all memory, leaving room for fragmentation
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import jax
import keras_hub
import tensorflow as tf
import tensorflow_datasets as tfds
import keras

# ✅ CRITICAL: Mixed precision is required for 9B models
keras.config.set_dtype_policy("mixed_bfloat16")

# Hide TF GPUs so JAX takes full control
tf.config.set_visible_devices([], "GPU")

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- JAX Device Detection ---
try:
    devices = jax.devices()
    # On TPU, we want the TPU devices, not CPU
    tpu_devices = [d for d in devices if d.platform == "tpu"]
    
    if not tpu_devices:
        TARGET_DEVICES = devices
    else:
        TARGET_DEVICES = tpu_devices

    DEVICES_AVAILABLE = len(TARGET_DEVICES)
    TARGET_WORLD_SIZE = DEVICES_AVAILABLE 
    logger.info(f"Targeting {TARGET_WORLD_SIZE} devices: {[str(d) for d in TARGET_DEVICES]}")

except Exception as e:
    logger.error(f"Could not initialize JAX. Error: {e}")
    TARGET_WORLD_SIZE = 0

from keras.src.distribution.tensor_parallel.tensor_parallel_keras import TensorParallelKeras

# --- Constants ---
# 🟢 FIX: Reduced Batch Size to 1 to fit Optimizer State in Memory
BATCH_SIZE = 1
SEQUENCE_LENGTH = 128
LEARNING_RATE = 1e-4
EPOCHS = 1 
STEPS_PER_EPOCH = 10 
VALIDATION_STEPS = 5

MODEL_MAPPING = {
    "gemma2_9b_en": keras_hub.models.GemmaCausalLM, 
}

# ----------------------------------------------------------------------
# --- Dataset Helpers ---
# ----------------------------------------------------------------------

def load_shakespeare_dataset(model_preset):
    logger.info(f"Loading tokenizer for {model_preset}...")
    tokenizer = keras_hub.models.GemmaTokenizer.from_preset(model_preset)

    logger.info("Loading Tiny Shakespeare dataset...")
    ds = tfds.load("tiny_shakespeare", split="train", as_supervised=False)
    ds = ds.take(100) # Small subset for verification
    
    text = "".join(example["text"].decode("utf-8") for example in ds.as_numpy_iterator())
    token_ids = tokenizer.tokenize(text)

    num_tokens = (len(token_ids) // (SEQUENCE_LENGTH + 1)) * (SEQUENCE_LENGTH + 1)
    sequences = np.array(token_ids[:num_tokens]).reshape(-1, SEQUENCE_LENGTH + 1)

    all_data = tf.data.Dataset.from_tensor_slices(sequences)
    
    # Ensure enough data
    if sequences.shape[0] < (BATCH_SIZE * (STEPS_PER_EPOCH + VALIDATION_STEPS)):
        all_data = all_data.repeat(20)

    num_train_samples = int(0.9 * sequences.shape[0])
    train_ds = all_data.take(num_train_samples)
    val_ds = all_data.skip(num_train_samples)

    return train_ds, val_ds

def format_for_causal_lm(data):
    """Formats data for KerasNLP's CausalLM."""
    # 🟢 FIX: Return simple tuple (x, y). 
    # Dropping 'padding_mask' prevents the JAX 'Abstract Tracer' error
    # and the 'Structure Mismatch' error.
    x = data[:-1]
    y = data[1:]
    return x, y

# ----------------------------------------------------------------------
# --- Main Verification ---
# ----------------------------------------------------------------------

def run_model_verification(preset_name, model_class):
    gc.collect()
    try: jax.clear_backends() 
    except: pass

    logger.info(f"--- VERIFICATION FOR: {preset_name.upper()} ---")

    train_ds_raw, val_ds_raw = load_shakespeare_dataset(preset_name)

    train_ds = (
        train_ds_raw
        .map(format_for_causal_lm, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
        .repeat()
    )
    val_ds = (
        val_ds_raw
        .map(format_for_causal_lm, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
        .repeat()
    )

    logger.info("\n--- Initializing Tensor Parallel (TP) Model ---")
    
    # 1. Get CPU device for the "Master" copy
    cpu_device = jax.devices("cpu")[0]

    # 2. Define builder with CPU forcing
    def model_builder():
        # Force allocation on CPU RAM first
        with jax.default_device(cpu_device):
            return model_class.from_preset(
                preset_name, 
                load_weights=False  # Random weights for verification
            )

    # 3. Instantiate TP Keras
    tp_model = TensorParallelKeras(
        model_input=model_builder,
        device_count=TARGET_WORLD_SIZE,
        device_ids=TARGET_DEVICES,
    )

    # 4. Compile
    tp_model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=LEARNING_RATE),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        jit_compile=True 
    )

    logger.info(f"Starting training with Batch Size: {BATCH_SIZE}")

    # 5. Fit
    tp_history = tp_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_steps=VALIDATION_STEPS,
        verbose=1,
    )
    
    logger.info("✅ SUCCESS: TP model training finished without errors.")
    return True

if __name__ == "__main__":
    if TARGET_WORLD_SIZE == 0:
        sys.exit(1)

    for preset, model_class in MODEL_MAPPING.items():
        try:
            run_model_verification(preset, model_class)
        except Exception as e:
            logger.error(f"Test for {preset} failed: {e}", exc_info=True)