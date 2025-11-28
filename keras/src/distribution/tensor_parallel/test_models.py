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
# Ensure we have enough host device count if testing on CPU backend, 
# otherwise JAX defaults to 1.
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import jax
import keras_hub
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

import keras

# --- CRITICAL: MIXED PRECISION TO PREVENT OOM ---
# Gemma 9B in float32 is ~36GB. In bfloat16 it is ~18GB.
# This must be set before loading the model.
keras.config.set_dtype_policy("bfloat16")

tf.config.set_visible_devices([], "GPU")

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- JAX Device Detection ---
try:
    devices = jax.devices()
    logger.info(f"JAX devices found: {[str(d) for d in devices]}")
    host_devices = [d for d in devices if d.platform == "cpu"]
    if not host_devices:
        host_devices = devices

    # We want to use 2 devices for Tensor Parallelism
    TARGET_WORLD_SIZE = 2
    DEVICES_AVAILABLE = len(host_devices)

    if DEVICES_AVAILABLE < TARGET_WORLD_SIZE:
        logger.warning(
            f"Requested {TARGET_WORLD_SIZE} devices, but only {DEVICES_AVAILABLE} available."
        )
        TARGET_DEVICES = host_devices
        TARGET_WORLD_SIZE = DEVICES_AVAILABLE
    else:
        TARGET_DEVICES = host_devices[:TARGET_WORLD_SIZE]
        logger.info(
            f"Targeting devices for parallelism: {[str(d) for d in TARGET_DEVICES]}"
        )

except Exception as e:
    logger.error(f"Could not initialize JAX. Error: {e}")
    TARGET_WORLD_SIZE = 0

from keras.src.distribution.tensor_parallel.tensor_parallel_keras import (
    TensorParallelKeras,
)

# --- Constants ---
BATCH_SIZE = 1  # Reduced batch size for 9B model
SEQUENCE_LENGTH = 32 # Reduced seq len for quick verification
LEARNING_RATE = 5e-5
EPOCHS = 1
STEPS_PER_EPOCH = 5
VALIDATION_STEPS = 2

# Using Gemma 2 9B
MODEL_PRESET = "gemma2_9b_en" 

# ----------------------------------------------------------------------
# --- Dataset and Model Helpers ---
# ----------------------------------------------------------------------

def load_dataset(model_preset):
    """Loads dataset and tokenizes using the Gemma tokenizer."""
    logger.info(f"Loading Tiny Shakespeare for {model_preset}...")
    
    # Load raw text
    ds = tfds.load("tiny_shakespeare", split="train", as_supervised=False)
    text = "".join(example["text"].decode("utf-8") for example in ds.as_numpy_iterator())
    
    # Load Tokenizer associated with the preset
    tokenizer = keras_hub.models.GemmaTokenizer.from_preset(model_preset)
    
    # Tokenize (returns a tensor of IDs)
    # Note: Shorten text for quick tokenization in this demo
    short_text = text[:100000] 
    token_ids = tokenizer(short_text)
    
    # If output is a dict (some versions), extract ids
    if isinstance(token_ids, dict):
        token_ids = token_ids["token_ids"]
        
    token_ids = np.array(token_ids)

    # Create sequences
    num_tokens = (len(token_ids) // (SEQUENCE_LENGTH + 1)) * (SEQUENCE_LENGTH + 1)
    sequences = token_ids[:num_tokens].reshape(-1, SEQUENCE_LENGTH + 1)
    
    # Create TF Dataset
    all_data = tf.data.Dataset.from_tensor_slices(sequences)
    
    num_sequences = sequences.shape[0]
    num_train_samples = int(0.9 * num_sequences)
    
    train_ds = all_data.take(num_train_samples)
    val_ds = all_data.skip(num_train_samples)
    
    logger.info(f"Dataset ready: {num_train_samples} train sequences.")
    return train_ds, val_ds

def format_for_causal_lm(data):
    """Formats data for CausalLM."""
    # Inputs: tokens [0:-1], Labels: tokens [1:]
    return (
        {
            "token_ids": data[:-1],
            "padding_mask": tf.ones_like(data[:-1], dtype=tf.bool),
        },
        data[1:]
    )

def run_verification():
    if TARGET_WORLD_SIZE < 2:
        logger.error("Need at least 2 devices for TP verification.")
        return

    # 1. Load Data
    train_ds_raw, val_ds_raw = load_dataset(MODEL_PRESET)
    
    train_ds = (
        train_ds_raw.batch(BATCH_SIZE, drop_remainder=True)
        .map(format_for_causal_lm, num_parallel_calls=tf.data.AUTOTUNE)
        .repeat()
    )
    val_ds = (
        val_ds_raw.batch(BATCH_SIZE, drop_remainder=True)
        .map(format_for_causal_lm, num_parallel_calls=tf.data.AUTOTUNE)
    )

    # 2. Load Master Model (CPU)
    # Using low_cpu_mem_usage logic implicit in Keras 3 with bfloat16
    logger.info(f"Loading master model: {MODEL_PRESET}...")
    master_model = keras_hub.models.GemmaCausalLM.from_preset(MODEL_PRESET)
    
    # 3. Create Tensor Parallel Model
    # The updated TensorParallelKeras will ingest the master, shard it, and delete the master from CPU.
    logger.info("Initializing Tensor Parallel Wrapper...")
    tp_model = TensorParallelKeras(
        model=master_model,
        device_count=TARGET_WORLD_SIZE,
        device_ids=TARGET_DEVICES,
    )

    # 4. Compile & Fit
    logger.info("Compiling...")
    tp_model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=LEARNING_RATE),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"], # Perplexity can be heavy, check simple accuracy first
    )

    logger.info("Starting Training...")
    tp_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_steps=VALIDATION_STEPS,
    )
    
    logger.info("✅ Verification Complete.")

if __name__ == "__main__":
    run_verification()