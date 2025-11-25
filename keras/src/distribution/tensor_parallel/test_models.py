import logging
import os
import sys
import time
import gc  # Added for memory management

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
    print(
        "Could not add project root to sys.path. "
        "Please run from the 'keras' directory or install as a package."
    )

# --- Backend and Device Configuration ---
os.environ["KERAS_BACKEND"] = "jax"

# ❌ REMOVED: Do not force CPU device count on TPU!
# os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import jax
import keras_hub
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

import keras

# ✅ ADDED: Essential for fitting 9B models on TPU v5e
keras.config.set_dtype_policy("mixed_bfloat16")

# Hide TF GPUs so JAX takes full control
tf.config.set_visible_devices([], "GPU")

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# --- JAX Device Detection ---
try:
    devices = jax.devices()
    logger.info(f"JAX devices found: {[str(d) for d in devices]}")
    
    # On TPU, we want the TPU devices, not CPU
    tpu_devices = [d for d in devices if d.platform == "tpu"]
    
    if not tpu_devices:
        logger.warning("No TPU devices found! Falling back to CPU/Emulator.")
        TARGET_DEVICES = devices
    else:
        TARGET_DEVICES = tpu_devices

    DEVICES_AVAILABLE = len(TARGET_DEVICES)
    # TPU v5e usually comes in slices of 4 or 8. Use whatever is available.
    TARGET_WORLD_SIZE = DEVICES_AVAILABLE 
    
    logger.info(
        f"Targeting {TARGET_WORLD_SIZE} devices for parallelism: "
        f"{[str(d) for d in TARGET_DEVICES]}"
    )

except Exception as e:
    logger.error(f"Could not initialize JAX or find devices. Error: {e}")
    TARGET_WORLD_SIZE = 0


from keras.src.distribution.tensor_parallel.tensor_parallel_keras import (
    TensorParallelKeras,
)

# --- Constants ---
# ⚠️ REDUCED BATCH SIZE: 16 might OOM on 9B model. Start small.
BATCH_SIZE = 4 
SEQUENCE_LENGTH = 128
LEARNING_RATE = 1e-4
EPOCHS = 1 # Reduced for verification speed
STEPS_PER_EPOCH = 5
VALIDATION_STEPS = 2

MODEL_MAPPING = {
    "gemma2_9b_en": keras_hub.models.GemmaCausalLM, 
}

# ----------------------------------------------------------------------
# --- Dataset and Model Helpers ---
# ----------------------------------------------------------------------

def load_shakespeare_dataset(model_preset):
    TOKENIZER_MAPPING = {
        "opt_125m_en": keras_hub.models.OPTTokenizer,
        "gemma_7b_en": keras_hub.models.GemmaTokenizer,
        "gemma2_9b_en": keras_hub.models.GemmaTokenizer,
    }

    logger.info(f"Loading tokenizer for {model_preset}...")
    
    # Robust tokenizer loading
    if model_preset in TOKENIZER_MAPPING:
        tokenizer_cls = TOKENIZER_MAPPING[model_preset]
        tokenizer = tokenizer_cls.from_preset(model_preset)
    else:
        tokenizer = keras_hub.models.GemmaTokenizer.from_preset(model_preset)

    logger.info("Loading Tiny Shakespeare dataset...")
    ds = tfds.load("tiny_shakespeare", split="train", as_supervised=False)
    
    # Take a small subset for verification to speed up tokenization
    ds = ds.take(100) 
    
    text = "".join(
        example["text"].decode("utf-8") for example in ds.as_numpy_iterator()
    )

    token_ids = tokenizer.tokenize(text)

    num_tokens = (len(token_ids) // (SEQUENCE_LENGTH + 1)) * (SEQUENCE_LENGTH + 1)
    sequences = np.array(token_ids[:num_tokens]).reshape(-1, SEQUENCE_LENGTH + 1)

    all_data = tf.data.Dataset.from_tensor_slices(sequences)

    num_sequences = sequences.shape[0]
    # Ensure we have enough data for the requested steps
    if num_sequences < (BATCH_SIZE * (STEPS_PER_EPOCH + VALIDATION_STEPS)):
        logger.warning("Dataset too small for requested steps. Repeating dataset.")
        all_data = all_data.repeat(10)

    num_train_samples = int(0.9 * num_sequences)
    train_ds = all_data.take(num_train_samples)
    val_ds = all_data.skip(num_train_samples)

    return train_ds, val_ds


def format_for_causal_lm(data):
    """Formats data for KerasNLP's CausalLM."""
    # Gemma expects dictionary inputs with these specific keys
    features = {
        "token_ids": data[:-1],
        "padding_mask": tf.ones_like(data[:-1], dtype=tf.bool),
    }
    labels = data[1:]
    return features, labels

# ----------------------------------------------------------------------
# --- Memory Helper ---
# ----------------------------------------------------------------------
def clear_memory():
    """Aggressively clears memory."""
    gc.collect()
    try:
        jax.clear_backends()
    except:
        pass
    logger.info("🧹 Memory cleared.")

# ----------------------------------------------------------------------
# --- Main Verification Function ---
# ----------------------------------------------------------------------

def run_model_verification(preset_name, model_class):
    clear_memory() # Clear before starting

    if TARGET_WORLD_SIZE < 2:
        logger.warning(
            f"SKIPPING {preset_name}: Need at least 2 devices, found {TARGET_WORLD_SIZE}"
        )
        return "SKIPPED"

    logger.info(f"--- VERIFICATION FOR: {preset_name.upper()} ---")

    # --- Dataset Pipeline ---
    train_ds_raw, val_ds_raw = load_shakespeare_dataset(preset_name)

    # Use map to apply the format function
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

    # --- Tensor Parallel Model Initialization ---
    logger.info("\n--- Initializing Tensor Parallel (TP) Model ---")
    
    # 1. Define the builder (Lazy Load)
    def model_builder():
        # load_weights=False is CRITICAL for 9B models to avoid host OOM
        # The weights will be initialized randomly on the device.
        return model_class.from_preset(
            preset_name, 
            load_weights=False 
        )

    # 2. Instantiate TP Keras
    tp_model = TensorParallelKeras(
        model_input=model_builder,
        device_count=TARGET_WORLD_SIZE,
        device_ids=TARGET_DEVICES,
    )

    # 3. Compile
    tp_model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=LEARNING_RATE),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        jit_compile=True # Ensure XLA compilation
    )

    logger.info(f"Starting training with Batch Size: {BATCH_SIZE}")

    # 4. Fit
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


# ----------------------------------------------------------------------
# --- Main Execution ---
# ----------------------------------------------------------------------

if __name__ == "__main__":
    if TARGET_WORLD_SIZE == 0:
        logger.critical("No JAX devices found. Aborting.")
        sys.exit(1)

    for preset, model_class in MODEL_MAPPING.items():
        try:
            run_model_verification(preset, model_class)
        except Exception as e:
            logger.error(f"Test for {preset} failed: {e}", exc_info=True)