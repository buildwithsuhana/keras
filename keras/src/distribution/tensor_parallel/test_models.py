import os

# --- CRITICAL: Memory Management Settings ---
# prevent JAX from pre-allocating all VRAM, allowing gradual growth
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

# Force JAX backend
os.environ["KERAS_BACKEND"] = "jax"
# Ensure CPU host device count is sufficient if needed
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import logging
import sys
import gc
import time
import numpy as np

import jax
import keras
import keras_hub
import tensorflow as tf
import tensorflow_datasets as tfds

# --- Set Mixed Precision immediately ---
# bfloat16 reduces memory from ~36GB to ~18GB for Gemma 9B
keras.config.set_dtype_policy("bfloat16")

tf.config.set_visible_devices([], "GPU") # Hide GPUs from TF data pipeline to save VRAM

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Project Root Setup (if running from subfolder) ---
try:
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(_script_dir)))
    )
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
except Exception:
    pass

from keras.src.distribution.tensor_parallel.tensor_parallel_keras import (
    TensorParallelKeras,
)

# --- Configuration ---
MODEL_PRESET = "gemma2_9b_en"
BATCH_SIZE = 4       # Keep small for 9B model
SEQUENCE_LENGTH = 128
LEARNING_RATE = 5e-5
EPOCHS = 1
STEPS_PER_EPOCH = 5

def get_devices():
    try:
        devices = jax.devices()
        logger.info(f"Available JAX devices: {[str(d) for d in devices]}")
        
        # Filter for accelerators (GPUs/TPUs)
        accel_devices = [d for d in devices if d.platform != "cpu"]
        
        if len(accel_devices) >= 2:
            return 2, accel_devices[:2]
        else:
            logger.warning("Not enough GPUs found. Using CPU devices for testing logic (slow).")
            cpu_devices = [d for d in devices if d.platform == "cpu"]
            return 2, cpu_devices[:2]
            
    except Exception as e:
        logger.error(f"Device detection failed: {e}")
        return 0, []

def load_data(preset):
    logger.info("Loading Tiny Shakespeare dataset...")
    ds = tfds.load("tiny_shakespeare", split="train", as_supervised=False)
    text = "".join(ex["text"].decode("utf-8") for ex in ds.as_numpy_iterator())
    
    logger.info("Tokenizing...")
    tokenizer = keras_hub.models.GemmaTokenizer.from_preset(preset)
    
    # Tokenize a subset to be fast
    tokens = tokenizer(text[:50000])
    if isinstance(tokens, dict):
        tokens = tokens["token_ids"]
    tokens = np.array(tokens)
    
    # Create sequences
    total_tokens = (len(tokens) // (SEQUENCE_LENGTH + 1)) * (SEQUENCE_LENGTH + 1)
    tokens = tokens[:total_tokens].reshape(-1, SEQUENCE_LENGTH + 1)
    
    dataset = tf.data.Dataset.from_tensor_slices(tokens)
    
    def prepare_batch(batch):
        return (
            {
                "token_ids": batch[:-1],
                "padding_mask": tf.ones_like(batch[:-1], dtype=tf.bool)
            },
            batch[1:] # Labels are shifted by 1
        )

    dataset = (
        dataset
        .batch(BATCH_SIZE, drop_remainder=True)
        .map(prepare_batch, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )
    return dataset

def run_training():
    device_count, target_devices = get_devices()
    if device_count < 2:
        logger.error("Aborting: Need at least 2 devices.")
        return

    train_ds = load_data(MODEL_PRESET)

    # --- 1. Load Master Model on CPU (The Fix) ---
    logger.info(f"Loading {MODEL_PRESET} to System RAM (CPU)...")
    logger.info("This prevents GPU OOM before sharding begins.")
    
    # Force placement on CPU to hold the ~18GB weights
    with keras.device("cpu"):
        master_model = keras_hub.models.GemmaCausalLM.from_preset(MODEL_PRESET)
    
    logger.info("✅ Master model loaded on CPU.")

    # --- 2. Initialize Tensor Parallelism ---
    # This class will migrate weights Slice-by-Slice from CPU to GPUs
    logger.info("Starting Tensor Parallel Sharding...")
    tp_model = TensorParallelKeras(
        model=master_model,
        device_count=device_count,
        device_ids=[str(d) for d in target_devices]
    )
    
    # Master model is now deleted from CPU RAM inside TensorParallelKeras

    # --- 3. Compile and Train ---
    logger.info("Compiling model...")
    tp_model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=LEARNING_RATE),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )

    logger.info("Starting Training Loop...")
    tp_model.fit(
        train_ds,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH
    )
    
    logger.info("🎉 Training Finished Successfully!")

if __name__ == "__main__":
    run_training()