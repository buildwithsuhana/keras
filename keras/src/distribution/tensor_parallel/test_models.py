import logging
import os
import sys
import gc
import numpy as np

# --- 1. CONFIGURATION & ENV VARS ---
os.environ["KERAS_BACKEND"] = "jax"

# JAX Memory Management for GPUs
# 'false' = Allocate memory on demand (prevents OOM on startup)
# 'platform' = Use the system allocator
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

# --- 2. PROJECT ROOT SETUP ---
try:
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(_script_dir)))
    )
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
except Exception:
    print("Could not add project root to sys.path.")

# --- 3. IMPORT KERAS (MUST BE FIRST) ---
# Fixes the circular import 'AttributeError: ... has no attribute Operation'
import keras

# --- 4. IMPORT EXTENSIONS & BACKENDS ---
import keras_hub
import jax
import tensorflow as tf
import tensorflow_datasets as tfds

# --- 5. GPU & MEMORY SETUP ---
# Enable Mixed Precision (Mandatory for T4s to save VRAM)
keras.config.set_dtype_policy("mixed_bfloat16")

# Hide GPUs from TensorFlow so JAX gets exclusive access
tf.config.set_visible_devices([], "GPU")

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- JAX Device Detection ---
try:
    # This should list your 2x T4 GPUs
    devices = jax.devices()
    GPU_DEVICES = [d for d in devices if d.platform == "gpu"]
    
    if len(GPU_DEVICES) < 2:
        logger.warning(f"⚠️ Warning: Found {len(GPU_DEVICES)} GPUs. Needs 2 for proper testing.")
        TARGET_DEVICES = devices # Fallback
    else:
        TARGET_DEVICES = GPU_DEVICES
        
    TARGET_WORLD_SIZE = len(TARGET_DEVICES)
    logger.info(f"✅ Targeting {TARGET_WORLD_SIZE} Devices: {TARGET_DEVICES}")

except Exception as e:
    logger.error(f"Could not initialize JAX. Error: {e}")
    sys.exit(1)

from keras.src.distribution.tensor_parallel.tensor_parallel_keras import TensorParallelKeras

# --- Constants for Gemma 9B on T4 ---
BATCH_SIZE = 1  # Keep at 1 to fit in 15GB VRAM
SEQUENCE_LENGTH = 128
LEARNING_RATE = 1e-4
EPOCHS = 1
STEPS_PER_EPOCH = 5
VALIDATION_STEPS = 2

MODEL_MAPPING = {
    "gemma2_2b_en": keras_hub.models.GemmaCausalLM, 
}

# ----------------------------------------------------------------------
# --- Custom Callback for Memory Management ---
# ----------------------------------------------------------------------
class ClearMemoryCallback(keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        # Clear Python garbage and JAX cache to protect 30GB System RAM
        gc.collect()
        jax.clear_caches()

# ----------------------------------------------------------------------
# --- Dataset Helpers ---
# ----------------------------------------------------------------------
def load_shakespeare_dataset(model_preset):
    logger.info(f"Loading tokenizer for {model_preset}...")
    tokenizer = keras_hub.models.GemmaTokenizer.from_preset(model_preset)

    logger.info("Loading Tiny Shakespeare dataset...")
    ds = tfds.load("tiny_shakespeare", split="train", as_supervised=False).take(50)
    
    text = "".join(example["text"].decode("utf-8") for example in ds.as_numpy_iterator())
    token_ids = tokenizer.tokenize(text)

    num_tokens = (len(token_ids) // (SEQUENCE_LENGTH + 1)) * (SEQUENCE_LENGTH + 1)
    sequences = np.array(token_ids[:num_tokens]).reshape(-1, SEQUENCE_LENGTH + 1)

    all_data = tf.data.Dataset.from_tensor_slices(sequences)
    
    if sequences.shape[0] < (BATCH_SIZE * (STEPS_PER_EPOCH + VALIDATION_STEPS)):
        all_data = all_data.repeat(20)

    num_train_samples = int(0.9 * sequences.shape[0])
    train_ds = all_data.take(num_train_samples)
    val_ds = all_data.skip(num_train_samples)

    return train_ds, val_ds

def format_for_causal_lm(data):
    x = data[:-1]
    y = data[1:]
    padding_mask = np.ones(x.shape, dtype=np.int32)
    x_dict = {"token_ids": x, "padding_mask": padding_mask}
    return x_dict, y

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
    )
    val_ds = (
        val_ds_raw
        .map(format_for_causal_lm, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )

    logger.info("\n--- Initializing Tensor Parallel (TP) Model ---")
    
    # Use CPU for initial definition structure (cheap)
    cpu_device = jax.devices("cpu")[0]

    def model_builder():
        # Load on CPU first, but keep weights empty (random)
        # to avoid OOMing your 30GB System RAM
        with jax.default_device(cpu_device):
            return model_class.from_preset(
                preset_name, 
                load_weights=False 
            )

    tp_model = TensorParallelKeras(
        model_input=model_builder,
        device_count=TARGET_WORLD_SIZE,
        device_ids=TARGET_DEVICES,
    )

    # 🟢 CRITICAL: SGD Optimizer
    # Adam requires Moment+Velocity (3x memory). SGD is 1x.
    # This is the only way 9B fits on 15GB GPUs without LoRA.
    optimizer = keras.optimizers.SGD(learning_rate=LEARNING_RATE)

    tp_model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        jit_compile=True 
    )

    logger.info(f"Starting training with Batch Size: {BATCH_SIZE} on {TARGET_WORLD_SIZE} GPUs")

    tp_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_steps=VALIDATION_STEPS,
        verbose=1,
        callbacks=[ClearMemoryCallback()]
    )
    
    logger.info("✅ SUCCESS: TP model training finished.")
    return True

if __name__ == "__main__":
    for preset, model_class in MODEL_MAPPING.items():
        try:
            run_model_verification(preset, model_class)
        except Exception as e:
            logger.error(f"Test for {preset} failed: {e}", exc_info=True)