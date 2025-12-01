import os
import gc
import shutil
import logging
import numpy as np

# --- 1. Aggressive Memory Environment Variables ---
# We limit JAX to 90% memory to prevent fragmentation, 
# and force it to act like a CPU allocator (slower but safer for OOM).
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90" 
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import jax
import keras
import keras_hub
import tensorflow as tf
import tensorflow_datasets as tfds

# Strict bfloat16 policy
keras.config.set_dtype_policy("bfloat16")
tf.config.set_visible_devices([], "GPU")

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from keras.src.distribution.tensor_parallel.tensor_parallel_keras import TensorParallelKeras

# --- CONFIGURATION ---
MODEL_PRESET = "gemma2_9b_en"
BATCH_SIZE = 1
# REDUCED: 128 -> 64 to save Activation Memory
SEQUENCE_LENGTH = 128
LEARNING_RATE = 1e-4 # Slightly higher for SGD
EPOCHS = 1
STEPS_PER_EPOCH = 5

def get_devices():
    devices = jax.devices()
    accel_devices = [d for d in devices if d.platform != "cpu"]
    return (len(accel_devices), accel_devices) if accel_devices else (0, [])

def load_data(preset):
    logger.info("Loading Data...")
    ds = tfds.load("tiny_shakespeare", split="train", as_supervised=False)
    text = "".join(ex["text"].decode("utf-8") for ex in ds.as_numpy_iterator())
    
    tokenizer = keras_hub.models.GemmaTokenizer.from_preset(preset)
    tokens = tokenizer(text[:10000]) # Smaller subset
    if isinstance(tokens, dict): tokens = tokens["token_ids"]
    tokens = np.array(tokens)
    
    total_tokens = (len(tokens) // (SEQUENCE_LENGTH + 1)) * (SEQUENCE_LENGTH + 1)
    tokens = tokens[:total_tokens].reshape(-1, SEQUENCE_LENGTH + 1)
    
    dataset = tf.data.Dataset.from_tensor_slices(tokens)
    def prepare_batch(batch):
        return ({"token_ids": batch[:-1], "padding_mask": tf.ones_like(batch[:-1], dtype=tf.bool)}, batch[1:])

    return dataset.batch(BATCH_SIZE, drop_remainder=True)

def model_factory():
    logger.info(f"🏭 Factory: Loading {MODEL_PRESET}...")
    with keras.device("cpu"):
        model = keras_hub.models.GemmaCausalLM.from_preset(MODEL_PRESET)
        return model

def run_training():
    device_count, target_devices = get_devices()
    if device_count < 2:
        logger.error("Need 2 GPUs.")
        return

    # Clear memory before we start
    gc.collect()
    jax.clear_caches()

    train_ds = load_data(MODEL_PRESET)

    logger.info("Preparing Tensor Parallel Model...")
    tp_model = TensorParallelKeras(
        model=model_factory, 
        device_count=device_count,
        device_ids=[str(d) for d in target_devices]
    )
    
    logger.info("Compiling model with SGD...")
    # --- CRITICAL: USE SGD ---
    # AdamW stores 2 extra variables per weight (massive memory usage).
    # SGD stores 0 extra variables.
    optimizer = keras.optimizers.SGD(
        learning_rate=LEARNING_RATE,
        momentum=0.9
    )

    tp_model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )

    logger.info("Starting Training Loop...")
    try:
        tp_model.fit(train_ds, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH)
        logger.info("🎉 Success!")
    except Exception as e:
        logger.error(f"Training Failed: {e}")
        # If it fails, print memory stats (if available)
        try:
            logger.info(jax.local_devices()[0].memory_stats())
        except:
            pass
        raise e

    if hasattr(tp_model, 'temp_dir') and os.path.exists(tp_model.temp_dir):
        shutil.rmtree(tp_model.temp_dir)

if __name__ == "__main__":
    run_training()