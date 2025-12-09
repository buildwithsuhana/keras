import os
import gc
import shutil
import logging
import time
import subprocess
import numpy as np

# --- 1. Environment Setup ---
# Clear potential XLA flags that force CPU
if "XLA_FLAGS" in os.environ: del os.environ["XLA_FLAGS"]

# Set Keras to use JAX backend
os.environ["KERAS_BACKEND"] = "jax"

# Critical: Prevent JAX from pre-allocating all VRAM so we can track growth
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
# Use mixed precision for memory efficiency
keras.config.set_dtype_policy("bfloat16")

# Hide GPUs from TensorFlow so JAX can have exclusive access
tf.config.set_visible_devices([], "GPU")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- 3. Patching for Tensor Parallelism ---
try:
    # Patch dot_product_attention to handle sharded inputs where head counts might not match
    import keras.src.backend.jax.nn as jax_keras_nn
    def safe_dot_product_attention(query, key, value, bias=None, mask=None, scale=None, is_causal=False, **kwargs):
        q_heads = query.shape[-2]
        k_heads = key.shape[-2]
        # If query heads don't match key heads (due to sharding splits), repeat keys/values
        if q_heads != k_heads:
            rep_factor = q_heads // k_heads
            key = jnp.repeat(key, rep_factor, axis=-2)
            value = jnp.repeat(value, rep_factor, axis=-2)
        return jax.nn.dot_product_attention(query, key, value, bias=bias, mask=None, scale=scale, is_causal=is_causal)
    jax_keras_nn.dot_product_attention = safe_dot_product_attention
except: 
    pass

# Import your custom TP implementation
from keras.src.distribution.tensor_parallel.tensor_parallel_keras import TensorParallelKeras

# --- 4. Training Configuration ---
MODEL_PRESET = "gemma2_9b_en"
BATCH_SIZE = 1  # Low batch size for large model on T4s
SEQUENCE_LENGTH = 128
LEARNING_RATE = 1e-4 
EPOCHS = 1
STEPS_PER_EPOCH = 5

# --- 5. Helper Utilities ---

def get_devices():
    """Detects available JAX accelerators."""
    devices = jax.devices()
    accel = [d for d in devices if d.platform != "cpu"]
    return (len(accel), accel) if accel else (0, [])

def log_gpu_memory(stage=""):
    """Queries nvidia-smi to log exact VRAM usage for all GPUs."""
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
        logger.warning(f"Could not retrieve GPU stats: {e}")

def inspect_shards(tp_model, devices):
    """Verifies that shard variables are physically located on the correct devices."""
    logger.info("🕵️ INSPECTING SHARD PLACEMENT...")
    for i, shard in enumerate(tp_model.model_shards):
        expect = str(devices[i])
        # Grab the first trainable variable to check its device
        if not shard.trainable_variables:
            logger.warning(f"   Shard {i} has no trainable variables.")
            continue
            
        var = shard.trainable_variables[0]
        try: 
            # JAX variables often have a .device or .devices() attribute
            if hasattr(var, 'value'):
                actual = str(list(var.value.devices())[0])
            else:
                actual = str(var.device)
        except: 
            actual = "Unknown"
        
        # Simple string matching to confirm device alignment
        is_match = (str(i) in actual) or (expect in actual)
        status = "✅ OK" if is_match else "❌ WRONG DEVICE"
        logger.info(f"   Shard {i} | Expect: {expect} | Actual: {actual} | {status}")

def load_data(preset):
    """Loads and tokenizes the Tiny Shakespeare dataset."""
    logger.info("Loading Data...")
    ds = tfds.load("tiny_shakespeare", split="train", as_supervised=False)
    text = "".join(ex["text"].decode("utf-8") for ex in ds.as_numpy_iterator())
    
    tokenizer = keras_hub.models.GemmaTokenizer.from_preset(preset)
    
    # Tokenize a subset to keep it fast
    tokens = tokenizer(text[:10000]) 
    if isinstance(tokens, dict): tokens = tokens["token_ids"]
    tokens = np.array(tokens)
    
    # Batching logic
    total = (len(tokens) // (SEQUENCE_LENGTH + 1)) * (SEQUENCE_LENGTH + 1)
    tokens = tokens[:total].reshape(-1, SEQUENCE_LENGTH + 1)
    dataset = tf.data.Dataset.from_tensor_slices(tokens)
    
    def prepare(b): 
        # Create (inputs, labels) pairs. Inputs need token_ids and padding_mask.
        return ({"token_ids": b[:-1], "padding_mask": tf.ones_like(b[:-1], dtype="int32")}, b[1:])
    
    return dataset.map(prepare, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE, drop_remainder=True)

def model_factory():
    """Factory to create the model on CPU first to save VRAM."""
    logger.info(f"🏭 Factory: Loading {MODEL_PRESET}...")
    with keras.device("cpu"):
        return keras_hub.models.GemmaCausalLM.from_preset(MODEL_PRESET)

# --- 6. Main Execution ---

def run_training():
    log_gpu_memory("Startup")
    count, devices = get_devices()
    logger.info(f"Detected {count} accelerators: {devices}")
    
    if count < 2: 
        logger.error("❌ Need 2+ accelerators for Tensor Parallelism.")
        return

    # Clean up any previous runs
    gc.collect()
    
    # 1. Load Data
    train_ds = load_data(MODEL_PRESET)
    log_gpu_memory("After Data Load")
    
    # 2. Initialize Distributed Model
    logger.info("Init TP Model...")
    dev_ids = [f"{d.platform}:{d.id}" for d in devices]
    logger.info(f"Using devices: {dev_ids}")
    
    # This call triggers the sharding logic inside TensorParallelKeras
    tp_model = TensorParallelKeras(model=model_factory, device_count=count, device_ids=dev_ids)
    
    log_gpu_memory("After TP Model Creation")
    inspect_shards(tp_model, devices)

    # 3. Build (Forward Pass with dummy data to initialize lazy layers)
    logger.info("Building...")
    dummy_input = {
        "token_ids": np.zeros((BATCH_SIZE, SEQUENCE_LENGTH), "int32"), 
        "padding_mask": np.ones((BATCH_SIZE, SEQUENCE_LENGTH), "int32")
    }
    tp_model(dummy_input)
    log_gpu_memory("After Build")

    # 4. Compile
    logger.info("Compiling...")
    tp_model.compile(
        optimizer=keras.optimizers.SGD(LEARNING_RATE, momentum=0.0),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
        # JIT must be True for JAX performance, Eager must be False to avoid Python loop overhead
        run_eagerly=False,  
        jit_compile=True    
    )
    log_gpu_memory("After Compile")

    # 5. Training Loop
    logger.info("Training...")
    start_time = time.time()
    history = tp_model.fit(train_ds, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH)
    end_time = time.time()
    
    logger.info(f"🎉 Training Success! Time: {end_time - start_time:.2f}s")
    log_gpu_memory("Final State")
    
    # Cleanup
    if hasattr(tp_model, 'temp_dir') and os.path.exists(tp_model.temp_dir):
        shutil.rmtree(tp_model.temp_dir)

if __name__ == "__main__":
    run_training()