import os
import gc
import shutil
import logging
import time
import subprocess
import numpy as np

# --- 1. Environment Setup ---
if "XLA_FLAGS" in os.environ: del os.environ["XLA_FLAGS"]
os.environ["KERAS_BACKEND"] = "jax"
# Prevent JAX from pre-allocating all VRAM
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

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
MODEL_PRESET = "gemma2_9b_en"
BATCH_SIZE = 1 
SEQUENCE_LENGTH = 128
LEARNING_RATE = 1e-4 
EPOCHS = 1
STEPS_PER_EPOCH = 5

# --- 5. Utilities ---
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
    except Exception as e:
        pass

def inspect_shards(tp_model, devices):
    logger.info("🕵️ INSPECTING SHARD PLACEMENT...")
    for i, shard in enumerate(tp_model.model_shards):
        expect = str(devices[i]).lower()
        if not shard.trainable_variables:
            continue
            
        # Check first variable
        var = shard.trainable_variables[0]
        actual = "Unknown"
        
        # JAX Variable Handling
        try:
            # For JAX, value is a jax.Array which has a .device() or .sharding
            val = var.value
            if hasattr(val, 'device'):
                actual = str(val.device()).lower()
            elif hasattr(val, 'devices'):
                 actual = str(list(val.devices())[0]).lower()
            elif hasattr(val, 'sharding'):
                 actual = str(list(val.sharding.device_set)[0]).lower()
        except:
            actual = str(var.device).lower()

        # Extract device ID (e.g. "gpu:0" or "cuda:0")
        is_match = False
        if "gpu" in expect and "gpu" in actual:
             # Check if index matches
             idx_expect = expect.split(':')[-1].strip()
             idx_actual = actual.split(':')[-1].strip()
             is_match = (idx_expect == idx_actual)
        
        status = "✅ OK" if is_match else f"❌ WRONG DEVICE (Got {actual})"
        logger.info(f"   Shard {i} | Expect: {expect} | Status: {status}")

def load_data(preset):
    logger.info("Loading Data...")
    ds = tfds.load("tiny_shakespeare", split="train", as_supervised=False)
    text = "".join(ex["text"].decode("utf-8") for ex in ds.as_numpy_iterator())
    tokenizer = keras_hub.models.GemmaTokenizer.from_preset(preset)
    tokens = tokenizer(text[:10000]) 
    if isinstance(tokens, dict): tokens = tokens["token_ids"]
    tokens = np.array(tokens)
    total = (len(tokens) // (SEQUENCE_LENGTH + 1)) * (SEQUENCE_LENGTH + 1)
    tokens = tokens[:total].reshape(-1, SEQUENCE_LENGTH + 1)
    dataset = tf.data.Dataset.from_tensor_slices(tokens)
    def prepare(b): return ({"token_ids": b[:-1], "padding_mask": tf.ones_like(b[:-1], dtype="int32")}, b[1:])
    return dataset.map(prepare, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE, drop_remainder=True)

def model_factory():
    logger.info(f"🏭 Factory: Loading {MODEL_PRESET}...")
    with keras.device("cpu"):
        return keras_hub.models.GemmaCausalLM.from_preset(MODEL_PRESET)

# --- 6. Execution ---
def run_training():
    log_gpu_memory("Startup")
    count, devices = get_devices()
    logger.info(f"Detected {count} accelerators: {devices}")
    if count < 2: return

    gc.collect()
    train_ds = load_data(MODEL_PRESET)
    log_gpu_memory("After Data Load")
    
    logger.info("Init TP Model...")
    # Normalize device IDs for Keras
    dev_ids = [f"gpu:{i}" for i in range(count)] 
    logger.info(f"Using devices: {dev_ids}")
    
    tp_model = TensorParallelKeras(model=model_factory, device_count=count, device_ids=dev_ids)
    
    log_gpu_memory("After TP Creation")
    inspect_shards(tp_model, devices)

    logger.info("Building...")
    tp_model({"token_ids": np.zeros((BATCH_SIZE, SEQUENCE_LENGTH), "int32"), "padding_mask": np.ones((BATCH_SIZE, SEQUENCE_LENGTH), "int32")})
    log_gpu_memory("After Build")

    logger.info("Compiling...")
    tp_model.compile(
        optimizer=keras.optimizers.SGD(LEARNING_RATE),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
        run_eagerly=False,
        jit_compile=True
    )

    logger.info("Training...")
    tp_model.fit(train_ds, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH)
    logger.info("🎉 Success!")
    
    if hasattr(tp_model, 'temp_dir'): shutil.rmtree(tp_model.temp_dir)

if __name__ == "__main__":
    run_training()