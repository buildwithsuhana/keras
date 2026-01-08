import os
import gc
import shutil
import logging
import time
import subprocess
import numpy as np
import ctypes

# --- 1. Environment Setup ---
if "XLA_FLAGS" in os.environ: del os.environ["XLA_FLAGS"]
os.environ["KERAS_BACKEND"] = "jax"
# Prevent XLA from reserving all GPU memory at once
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
# Limit compilation threads to prevent CPU RAM spikes during JIT
os.environ["XLA_FLAGS"] = "--xla_gpu_force_compilation_parallelism=1"

import jax
import jax.nn
import jax.numpy as jnp
import keras
import keras_hub
import tensorflow as tf
import tensorflow_datasets as tfds

# --- 2. GPU/TPU Configuration ---
keras.config.set_dtype_policy("bfloat16")
tf.config.set_visible_devices([], "GPU") 

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- 3. Patching ---
try:
    import keras.src.backend.jax.nn as jax_keras_nn
    def safe_dot_product_attention(query, key, value, bias=None, mask=None, scale=None, is_causal=False, **kwargs):
        q_heads = query.shape[-2]
        k_heads = key.shape[-2]
        if q_heads != k_heads:
            rep_factor = q_heads // k_heads
            key = jnp.repeat(key, rep_factor, axis=-2)
            value = jnp.repeat(value, rep_factor, axis=-2)
        return jax.nn.dot_product_attention(query, key, value, bias=bias, mask=None, scale=scale, is_causal=is_causal)
    jax_keras_nn.dot_product_attention = safe_dot_product_attention
except: 
    pass

from keras.src.distribution.tensor_parallel.tensor_parallel_keras import TensorParallelKeras

# --- 4. Config ---
MODEL_PRESET = "gemma_1b_en"
BATCH_SIZE = 1 
SEQUENCE_LENGTH = 128
LEARNING_RATE = 1e-4 
EPOCHS = 1
STEPS_PER_EPOCH = 5

# --- 5. Utilities ---
def flush_ram():
    """Aggressively clear system RAM."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except:
        pass

def get_devices():
    devices = jax.devices()
    accel = [d for d in devices if d.platform != "cpu"]
    return (len(accel), accel) if accel else (0, [])

def log_gpu_memory(stage=""):
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
            encoding='utf-8'
        )
        mems = [int(x) for x in result.strip().split('\n') if x.strip()]
        log_str = f"🎮 [GPU VRAM] {stage} | "
        for i, mem in enumerate(mems):
            log_str += f"GPU{i}: {mem} MiB | "
        logger.info(log_str)
    except Exception:
        pass

def load_data(preset):
    logger.info("Loading Data (Streaming Mode)...")
    # Tiny shakespeare is small, but we use a pattern for larger datasets to save RAM
    ds = tfds.load("tiny_shakespeare", split="train", as_supervised=False)
    text = "".join(ex["text"].decode("utf-8") for ex in ds.as_numpy_iterator())
    tokenizer = keras_hub.models.GemmaTokenizer.from_preset(preset)
    tokens = tokenizer(text[:10000]) 
    if isinstance(tokens, dict): tokens = tokens["token_ids"]
    tokens = np.array(tokens)
    total = (len(tokens) // (SEQUENCE_LENGTH + 1)) * (SEQUENCE_LENGTH + 1)
    tokens = tokens[:total].reshape(-1, SEQUENCE_LENGTH + 1)
    dataset = tf.data.Dataset.from_tensor_slices(tokens)
    
    def prepare(b): 
        return ({"token_ids": b[:-1], "padding_mask": tf.ones_like(b[:-1], dtype="int32")}, b[1:])
    
    return dataset.map(prepare, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE, drop_remainder=True)

def model_factory():
    """
    Lazy Factory: Forces the Keras backend to stay on CPU for the entire
    instantiation process of the 18GB Gemma model.
    """
    logger.info(f"🏭 Factory: Instantiating {MODEL_PRESET}...")
    
    # Force the backend to use CPU for ALL initializations in this scope
    with keras.utils.custom_object_scope(None):
        with keras.device("cpu"):
            # Use 'load_weights=False' initially if the Hub API supports it,
            # or rely on the fact that TensorParallelKeras will swap these values anyway.
            model = keras_hub.models.GemmaCausalLM.from_preset(
                MODEL_PRESET, 
                load_weights=True # Must be True to get the weights for offloading
            )
            return model

# In run_training(), add this JAX-specific environment variable 
# to prevent it from greedily taking the first GPU for construction
os.environ["JAX_PLATFORM_NAME"] = "cpu"

# --- 6. Execution ---
def run_training():
    log_gpu_memory("Startup")
    count, devices = get_devices()
    logger.info(f"Detected {count} accelerators.")
    
    if count < 2: 
        logger.error("Need at least 2 accelerators for Tensor Parallelism.")
        return

    # Clear RAM before loading anything heavy
    flush_ram()
    
    train_ds = load_data(MODEL_PRESET)
    log_gpu_memory("After Data Load")
    
    logger.info("Initializing TensorParallelKeras...")
    dev_ids = [f"gpu:{i}" for i in range(count)] 
    
    # The Model Factory is passed as a lambda to avoid immediate allocation
    tp_model = TensorParallelKeras(
        model=model_factory, 
        device_count=count, 
        device_ids=dev_ids
    )
    
    flush_ram()
    log_gpu_memory("After TP Creation")

    logger.info("Compiling with JIT (Serialized)...")
    tp_model.compile(
        optimizer=keras.optimizers.SGD(LEARNING_RATE),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
        run_eagerly=False,
        jit_compile=True
    )

    logger.info("Starting Training Fit...")
    # BATCH_SIZE=1 and tiny steps per epoch ensure we test the loop without hitting limits
    try:
        tp_model.fit(train_ds, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH)
        logger.info("🎉 Step Execution Success!")
    except Exception as e:
        logger.error(f"💥 Training failed: {e}")
    finally:
        if hasattr(tp_model, 'temp_dir') and os.path.exists(tp_model.temp_dir):
            shutil.rmtree(tp_model.temp_dir)

if __name__ == "__main__":
    run_training()