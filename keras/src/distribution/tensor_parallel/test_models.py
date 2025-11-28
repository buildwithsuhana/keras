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

# --- Backend and Device Configuration ---
os.environ["KERAS_BACKEND"] = "jax"
# Prevent JAX from pre-allocating all memory, leaving room for Keras loading
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import jax
import keras_hub
import tensorflow as tf
import tensorflow_datasets as tfds
import keras

# Hide GPUs from TensorFlow so JAX gets exclusive access
tf.config.set_visible_devices([], "GPU")

# [CRITICAL] Use mixed_bfloat16 to fit model in memory (18GB vs 36GB)
keras.config.set_dtype_policy("mixed_bfloat16")

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- JAX Device Detection ---
try:
    devices = jax.devices()
    accel_devices = [d for d in devices if d.platform != "cpu"]
    
    if accel_devices:
        # [FIX] Convert Device objects to standard Keras string format "gpu:0", "gpu:1"
        TARGET_DEVICES = [f"gpu:{d.id}" for d in accel_devices]
        WORLD_SIZE = len(TARGET_DEVICES)
        logger.info(f"🚀 Using Accelerators: {TARGET_DEVICES}")
    else:
        TARGET_DEVICES = ["cpu:0"]
        WORLD_SIZE = 1
        logger.warning("⚠️ No Accelerators found. Falling back to CPU.")

except Exception as e:
    logger.error(f"Could not initialize JAX. Error: {e}")
    TARGET_DEVICES = []
    WORLD_SIZE = 0

from keras.src.distribution.tensor_parallel.tensor_parallel_keras import (
    TensorParallelKeras,
)

# --- Constants ---
BATCH_SIZE = 1
SEQUENCE_LENGTH = 128
LEARNING_RATE = 5e-5
EPOCHS = 1
STEPS_PER_EPOCH = 5
VALIDATION_STEPS = 5

MODEL_MAPPING = {
    "gemma2_9b_en": keras_hub.models.GemmaCausalLM,
}

def load_shakespeare_dataset(model_preset):
    logger.info(f"Loading Tiny Shakespeare for {model_preset}...")
    ds = tfds.load("tiny_shakespeare", split="train", as_supervised=False)
    
    # Tiny slice for verification speed
    text = "".join(example["text"].decode("utf-8") for example in ds.as_numpy_iterator())
    text = text[:50000]

    # Load ONLY the preprocessor (lightweight)
    logger.info("   - Loading lightweight preprocessor...")
    if "gemma" in model_preset:
        preprocessor = keras_hub.models.GemmaCausalLMPreprocessor.from_preset(model_preset)
    else:
        preprocessor = keras_hub.models.OPTCausalLMPreprocessor.from_preset(model_preset)

    tokenizer = preprocessor.tokenizer
    token_ids = tokenizer(text)
    
    length = SEQUENCE_LENGTH + 1
    num_tokens = (len(token_ids) // length) * length
    token_ids = token_ids[:num_tokens]
    
    sequences = tf.data.Dataset.from_tensor_slices(
        np.array(token_ids).reshape(-1, length)
    )
    
    def split_input_label(x):
        return {
            "token_ids": x[:-1],
            "padding_mask": tf.ones_like(x[:-1], dtype=tf.bool),
        }, x[1:]

    sequences = sequences.map(split_input_label, num_parallel_calls=tf.data.AUTOTUNE)
    
    train_ds = sequences.take(50).batch(BATCH_SIZE).repeat()
    val_ds = sequences.skip(50).take(10).batch(BATCH_SIZE).repeat()
    return train_ds, val_ds

def run_model_verification(preset_name, model_class):
    gc.collect()
    logger.info(f"--- VERIFICATION FOR: {preset_name.upper()} ---")
    
    train_ds, val_ds = load_shakespeare_dataset(preset_name)

    logger.info("\n--- Training Tensor Parallel (TP) Model ---")
    
    # 1. Load Skeleton on CPU
    logger.info("⏳ Loading model skeleton onto CPU RAM...")
    with keras.device("cpu"):
        original_model = model_class.from_preset(
            preset_name,
            dtype="bfloat16",
        )
        if "gemma" in preset_name:
            logger.info("   - Enabling LoRA for memory efficiency...")
            original_model.backbone.enable_lora(rank=4)

    # 2. Distribute with Disk Offloading
    logger.info("   - Sharding model (Zero Stage Init)...")
    tp_model = TensorParallelKeras(
        model=original_model,
        device_count=WORLD_SIZE,
        device_ids=TARGET_DEVICES,
        low_memory_mode=True 
    )

    # Cleanup CPU model immediately
    del original_model
    gc.collect()

    # 3. Compile & Fit
    logger.info("   - Compiling...")
    tp_model.compile(
        optimizer=keras.optimizers.AdamW(LEARNING_RATE),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        jit_compile=True 
    )

    logger.info("   - Fitting...")
    tp_history = tp_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_steps=VALIDATION_STEPS,
    )
    
    logger.info("✅ SUCCESS: TP training complete.")
    del tp_model
    gc.collect()
    return True

if __name__ == "__main__":
    for preset, cls in MODEL_MAPPING.items():
        try:
            run_model_verification(preset, cls)
        except Exception as e:
            logger.error(f"Failed: {e}", exc_info=True)