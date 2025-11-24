import logging
import os
import sys
import time
import gc # <--- Added for memory management

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
    pass

# --- Backend and Device Configuration ---
os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

import jax
import keras_hub
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import keras

# --- CRITICAL: Mixed Precision is REQUIRED for Gemma 7B on T4s ---
# Without this, the model is ~34GB (float32) and won't fit even when sharded.
keras.config.set_dtype_policy("mixed_bfloat16") 

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

    DEVICES_AVAILABLE = len(host_devices)
    WORLD_SIZE = 2

    if DEVICES_AVAILABLE < WORLD_SIZE:
        logger.warning(
            f"Requested {WORLD_SIZE} devices, but only {DEVICES_AVAILABLE} available."
        )
        TARGET_DEVICES = host_devices
        TARGET_WORLD_SIZE = DEVICES_AVAILABLE
    else:
        TARGET_DEVICES = host_devices[:WORLD_SIZE]
        TARGET_WORLD_SIZE = WORLD_SIZE
        logger.info(
            f"Targeting the first {TARGET_WORLD_SIZE} devices for parallelism: "
            f"{[str(d) for d in TARGET_DEVICES]}"
        )

except Exception as e:
    logger.error(f"Could not initialize JAX or find devices. Error: {e}")
    TARGET_WORLD_SIZE = 0

from keras.src.distribution.distribution_lib import AutoTPDistribution, DeviceMesh

# --- Constants ---
BATCH_SIZE = 1  # Reduced to 1 for Gemma 7B stability
SEQUENCE_LENGTH = 64 
LEARNING_RATE = 5e-5 # Lower LR for larger model
EPOCHS = 1
STEPS_PER_EPOCH = 3
VALIDATION_STEPS = 2

# --- MODEL MAPPING ---
MODEL_MAPPING = {
    # "opt_125m_en": keras_hub.models.OPTCausalLM,
    "gemma_7b_en": keras_hub.models.GemmaCausalLM, 
}

# ----------------------------------------------------------------------
# --- Dataset and Model Helpers ---
# ----------------------------------------------------------------------

def load_shakespeare_dataset(model_preset):
    # (Same implementation)
    logger.info(f"Loading Tiny Shakespeare for {model_preset}...")
    ds = tfds.load("tiny_shakespeare", split="train", as_supervised=False)
    text = "".join(example["text"].decode("utf-8") for example in ds.as_numpy_iterator())
    tokenizer = keras_hub.models.GemmaCausalLM.from_preset(model_preset).preprocessor.tokenizer
    token_ids = tokenizer.tokenize(text)
    num_tokens = (len(token_ids) // (SEQUENCE_LENGTH + 1)) * (SEQUENCE_LENGTH + 1)
    sequences = np.array(token_ids[:num_tokens]).reshape(-1, SEQUENCE_LENGTH + 1)
    all_data = tf.data.Dataset.from_tensor_slices(sequences)
    return all_data.take(100), all_data.skip(100).take(20) # Reduced dataset for quick test

def format_for_causal_lm(data):
    """Formats data for KerasNLP's CausalLM, creating features and labels."""
    # Force a unique buffer for the input before slicing
    data = tf.identity(data)
    token_ids = tf.identity(data[:, :-1])
    padding_mask = tf.identity(tf.ones_like(data[:, :-1], dtype=tf.bool))
    labels = tf.identity(data[:, 1:])
    features = {
        "token_ids": tf.identity(token_ids),
        "padding_mask": tf.identity(padding_mask),
    }
    labels = tf.identity(labels)
    return features, labels


def get_model_from_preset(preset_name, model_class):
    """Creates a CausalLM model from a KerasNLP preset."""
    logger.info(f"Creating {preset_name} model from KerasNLP preset...")
    model = model_class.from_preset(preset_name, preprocessor=None)
    logger.info(f"Model created with {model.count_params():,} parameters.")
    return model

def plot_training_graphs(history, preset_name):
    # (Same implementation)
    pass

# ----------------------------------------------------------------------
# --- Main Verification Function (Updated for Gemma Memory) ---
# ----------------------------------------------------------------------

def run_model_verification(preset_name, model_class):
    """
    Runs a training execution test for a given model preset using
    AutoTPDistribution with the scope() context manager.
    """
    if TARGET_WORLD_SIZE < 2:
        logger.warning(
            f"SKIPPING {preset_name}: Need at least 2 devices for tensor "
            f"parallelism, found {TARGET_WORLD_SIZE}"
        )
        return "SKIPPED"

    logger.info(f"--- VERIFICATION FOR: {preset_name.upper()} ---")

    # 1. Dataset Setup
    train_ds_raw, val_ds_raw = load_shakespeare_dataset(preset_name)
    train_ds = (
        train_ds_raw.batch(BATCH_SIZE, drop_remainder=True)
        .map(format_for_causal_lm, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE).repeat()
    )
    val_ds = (
        val_ds_raw.batch(BATCH_SIZE, drop_remainder=True)
        .map(format_for_causal_lm, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE).repeat()
    )

    # 2. Device Mesh
    device_mesh = DeviceMesh(
        shape=(1, TARGET_WORLD_SIZE),
        axis_names=["data", "model"],
        devices=TARGET_DEVICES,
    )

    # 3. Instantiate Distribution & Template
    # CRITICAL: Template must be on CPU to avoid VRAM OOM
    logger.info("Creating model template on CPU...")
    with keras.device("cpu"):
        model_template = get_model_from_preset(preset_name, model_class)

    logger.info("Initializing AutoTPDistribution...")
    distribution = AutoTPDistribution(
        model=model_template,
        device_mesh=device_mesh,
        auto_shard_dataset=True
    )
    
    # 4. CRITICAL MEMORY CLEANUP
    # Gemma 7B template takes ~15-30GB RAM. We MUST delete it before training.
    logger.info("🗑️ Deleting CPU template to free System RAM...")
    del model_template
    gc.collect()
    jax.clear_caches()
    
    # Get wrapped model
    tp_model = distribution.model

    # 5. Compile and Fit
    logger.info("\n--- Entering Distribution Scope ---")
    with distribution.scope():
        logger.info("Compiling model...")
        # Using AdamW with low epsilon for float16 stability
        tp_model.compile(
            optimizer=keras.optimizers.AdamW(learning_rate=LEARNING_RATE, epsilon=1e-5),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[keras_hub.metrics.Perplexity(from_logits=True, name="perplexity")],
        )

    logger.info("Starting training loop...")
    tp_history = tp_model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            steps_per_epoch=STEPS_PER_EPOCH,
            validation_steps=VALIDATION_STEPS,
            verbose=1,
    )

    logger.info("--- Exited Distribution Scope ---")
    logger.info("✅ SUCCESS: Gemma 7B trained successfully.")
    return True

if __name__ == "__main__":
    # (Same main block)
    for preset, model_class in MODEL_MAPPING.items():
        run_model_verification(preset, model_class)