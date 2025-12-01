import os
import gc
import shutil
import logging
import numpy as np

# --- 1. Aggressive Memory Environment Variables ---
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90" 
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["KERAS_BACKEND"] = "jax"
# Ensure we see all TPUs
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import jax
import jax.nn
import jax.numpy as jnp
import keras
import keras_hub
import tensorflow as tf
import tensorflow_datasets as tfds

# Strict bfloat16 policy for TPU efficiency
keras.config.set_dtype_policy("bfloat16")
tf.config.set_visible_devices([], "GPU")

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- 🩹 ADVANCED MONKEY PATCH (Fixes GQA + Driver Mismatch) ---
try:
    import keras.src.backend.jax.nn as jax_keras_nn
    
    def safe_dot_product_attention(query, key, value, bias=None, mask=None, scale=None, is_causal=False, **kwargs):
        # 1. FIX SHAPE MISMATCH (GQA Support)
        # Gemma 2 uses Grouped Query Attention. 
        # If Query has 16 heads and Key has 8, we must repeat Key 2x to match.
        q_heads = query.shape[-2]
        k_heads = key.shape[-2]
        
        if q_heads != k_heads:
            rep_factor = q_heads // k_heads
            # Repeat the key/value heads to match query heads
            key = jnp.repeat(key, rep_factor, axis=-2)
            value = jnp.repeat(value, rep_factor, axis=-2)

        # 2. FIX DRIVER CRASH (ConcretizationTypeError)
        # We force mask=None because the "Pallas" kernel crashes on TPUs with old drivers
        # when it tries to read the mask. 'is_causal' handles the masking for us.
        return jax.nn.dot_product_attention(
            query, key, value, 
            bias=bias, 
            mask=None, 
            scale=scale, 
            is_causal=is_causal
        )
        
    jax_keras_nn.dot_product_attention = safe_dot_product_attention
    logger.info("🩹 Patch Applied: Enabled Manual GQA + Disabled Pallas Attention.")
except Exception as e:
    logger.warning(f"Failed to apply patch: {e}")
# ---------------------------------------------------------------

from keras.src.distribution.tensor_parallel.tensor_parallel_keras import TensorParallelKeras

# --- CONFIGURATION ---
MODEL_PRESET = "gemma2_9b_en"
BATCH_SIZE = 1 
SEQUENCE_LENGTH = 128
LEARNING_RATE = 1e-4 
EPOCHS = 1
STEPS_PER_EPOCH = 5

def get_devices():
    devices = jax.devices()
    # Filter for TPUs/GPUs only
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
    
    # Ensure tokens are split exactly into input+label blocks
    total_tokens = (len(tokens) // (SEQUENCE_LENGTH + 1)) * (SEQUENCE_LENGTH + 1)
    tokens = tokens[:total_tokens].reshape(-1, SEQUENCE_LENGTH + 1)
    
    dataset = tf.data.Dataset.from_tensor_slices(tokens)
    
    def prepare_batch(batch):
        # We keep the dictionary structure Keras expects
        return (
            {
                "token_ids": batch[:-1], 
                "padding_mask": tf.ones_like(batch[:-1], dtype="int32")
            }, 
            batch[1:]
        )

    dataset = dataset.map(prepare_batch, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(BATCH_SIZE, drop_remainder=True)

def model_factory():
    logger.info(f"🏭 Factory: Loading {MODEL_PRESET}...")
    # Load initial model on CPU to avoid OOM before sharding
    with keras.device("cpu"):
        model = keras_hub.models.GemmaCausalLM.from_preset(MODEL_PRESET)
        return model

def run_training():
    device_count, target_devices = get_devices()
    logger.info(f"Devices detected: {device_count}")
    
    if device_count < 2:
        logger.error("Need at least 2 accelerators for Tensor Parallelism.")
        return

    gc.collect()
    jax.clear_caches()

    train_ds = load_data(MODEL_PRESET)

    logger.info("Preparing Tensor Parallel Model...")
    tp_model = TensorParallelKeras(
        model=model_factory, 
        device_count=device_count,
        device_ids=[str(d) for d in target_devices]
    )

    logger.info("🔧 Manually building model to ensure variable initialization...")
    # CHANGE: Removed try/except block. If this fails, we want to crash NOW.
    dummy_inputs = {
        "token_ids": np.zeros((BATCH_SIZE, SEQUENCE_LENGTH), dtype="int32"),
        "padding_mask": np.ones((BATCH_SIZE, SEQUENCE_LENGTH), dtype="int32"),
    }
    tp_model(dummy_inputs)
    logger.info("✅ Model built successfully.")

    logger.info("Compiling model with SGD...")
    optimizer = keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=0.0)

    tp_model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
        jit_compile=False
    )
    logger.info("🕵️ Verifying Shard Placement...")
    for i, shard in enumerate(tp_model.model_shards):
        # Check the device of the first weight in each shard
        first_weight = shard.trainable_variables[0]
        logger.info(f"Shard {i} | Device: {first_weight.device} | Shape: {first_weight.shape}")

    logger.info("Starting Training Loop...")
    try:
        tp_model.fit(train_ds, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH)
        logger.info("🎉 Success!")
    except Exception as e:
        logger.error(f"Training Failed: {e}")
        try:
            # Attempt to print memory stats for debugging
            logger.info(jax.local_devices()[0].memory_stats())
        except:
            pass
        raise e
    finally:
        # Cleanup temp weights
        if hasattr(tp_model, 'temp_dir') and os.path.exists(tp_model.temp_dir):
            shutil.rmtree(tp_model.temp_dir)

if __name__ == "__main__":
    run_training()