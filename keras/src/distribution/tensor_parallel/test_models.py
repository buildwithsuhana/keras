import os
import gc
import shutil
import logging
import numpy as np

# --- 1. Aggressive Memory Environment Variables ---
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# Note: On TPUs, MEM_FRACTION behavior varies, but keeping it per your request.
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90" 
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["KERAS_BACKEND"] = "jax"
# Ensuring 8 devices are visible if using TPU/Multi-GPU
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import jax
import keras
import keras_hub
import tensorflow as tf
import tensorflow_datasets as tfds

# Strict bfloat16 policy for TPU/GPU memory savings
keras.config.set_dtype_policy("bfloat16")
tf.config.set_visible_devices([], "GPU")

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from keras.src.distribution.tensor_parallel.tensor_parallel_keras import TensorParallelKeras

# --- CONFIGURATION ---
MODEL_PRESET = "gemma2_9b_en"
BATCH_SIZE = 1 # Keep at 1 for 9B model on limited resources
SEQUENCE_LENGTH = 128
LEARNING_RATE = 1e-4 
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
    tokens = tokenizer(text[:10000]) 
    if isinstance(tokens, dict): tokens = tokens["token_ids"]
    tokens = np.array(tokens)
    
    # Ensure we have exact multiples of sequence length + 1 (for input+label)
    total_tokens = (len(tokens) // (SEQUENCE_LENGTH + 1)) * (SEQUENCE_LENGTH + 1)
    tokens = tokens[:total_tokens].reshape(-1, SEQUENCE_LENGTH + 1)
    
    dataset = tf.data.Dataset.from_tensor_slices(tokens)
    
    def prepare_batch(batch):
        # OPTIMIZATION: Removed "padding_mask". 
        # GemmaCausalLM handles unmasked input fine (assumes full attention).
        # This saves memory and complexity.
        return ({"token_ids": batch[:-1]}, batch[1:])

    # --- FIX 1: Apply the prepare_batch mapping ---
    dataset = dataset.map(prepare_batch, num_parallel_calls=tf.data.AUTOTUNE)
    
    return dataset.batch(BATCH_SIZE, drop_remainder=True)

def model_factory():
    logger.info(f"🏭 Factory: Loading {MODEL_PRESET}...")
    with keras.device("cpu"):
        model = keras_hub.models.GemmaCausalLM.from_preset(MODEL_PRESET)
        return model

def run_training():
    device_count, target_devices = get_devices()
    logger.info(f"Devices detected: {device_count}")
    
    if device_count < 2:
        logger.error("Need at least 2 accelerators for Tensor Parallelism.")
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

    # --- FIX 2: Manually Build the Model ---
    # This forces variable creation before the complex .fit() loop starts.
    logger.info("🔧 Manually building model to ensure variable initialization...")
    try:
        # Create dummy inputs matching (BATCH_SIZE, SEQUENCE_LENGTH)
        dummy_inputs = {
            "token_ids": np.zeros((BATCH_SIZE, SEQUENCE_LENGTH), dtype="int32"),
            # No padding mask needed as we removed it in load_data
        }
        # Run a single forward pass to initialize weights
        tp_model(dummy_inputs)
        logger.info("✅ Model built successfully.")
    except Exception as e:
        logger.warning(f"Build pass warning (ignore if training starts): {e}")

    logger.info("Compiling model with SGD...")
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
        try:
            # Print memory stats for the first device if available
            logger.info(jax.local_devices()[0].memory_stats())
        except:
            pass
        raise e

    if hasattr(tp_model, 'temp_dir') and os.path.exists(tp_model.temp_dir):
        shutil.rmtree(tp_model.temp_dir)

if __name__ == "__main__":
    run_training()