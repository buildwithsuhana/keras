import os
import gc
import shutil
import logging
import time
import numpy as np

if "XLA_FLAGS" in os.environ: del os.environ["XLA_FLAGS"]
os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import jax
import jax.nn
import jax.numpy as jnp
import keras
import keras_hub
import tensorflow as tf
import tensorflow_datasets as tfds

# TPU Setup
keras.config.set_dtype_policy("bfloat16")
tf.config.set_visible_devices([], "GPU")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Patching ---
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
except: pass

from keras.src.distribution.tensor_parallel.tensor_parallel_keras import TensorParallelKeras

# --- Config ---
MODEL_PRESET = "gemma2_9b_en"
BATCH_SIZE = 1 
SEQUENCE_LENGTH = 128
LEARNING_RATE = 1e-4 
EPOCHS = 1
STEPS_PER_EPOCH = 5

def get_devices():
    devices = jax.devices()
    accel = [d for d in devices if d.platform != "cpu"]
    return (len(accel), accel) if accel else (0, [])

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

def print_memory(tag):
    try:
        # Try to print for all local devices
        for d in jax.local_devices():
            if d.platform == "cpu": continue
            stats = d.memory_stats()
            used = stats['bytes_in_use'] / 1e9
            limit = stats['bytes_limit'] / 1e9
            logger.info(f"📊 MEMORY [{d}] @ {tag}: {used:.2f} / {limit:.2f} GB")
    except: pass

def inspect_shards(tp_model, devices):
    logger.info("🕵️ INSPECTING SHARD PLACEMENT...")
    for i, shard in enumerate(tp_model.model_shards):
        expect = str(devices[i])
        var = shard.trainable_variables[0]
        try: 
            actual = str(list(var.value.devices())[0]) if hasattr(var, 'value') else str(var.device)
        except: 
            actual = "Unknown"
        
        is_match = expect in actual or actual in expect
        status = "✅ OK" if is_match else "❌ WRONG DEVICE"
        logger.info(f"   Shard {i} | Expect: {expect} | Actual: {actual} | {status}")

def run_training():
    count, devices = get_devices()
    if count < 2: 
        logger.error("Need 2+ accelerators")
        return

    gc.collect()
    train_ds = load_data(MODEL_PRESET)
    
    logger.info("Init TP Model...")
    dev_ids = [f"{d.platform}:{d.id}" for d in devices]
    logger.info(f"Using devices: {dev_ids}")
    
    tp_model = TensorParallelKeras(model=model_factory, device_count=count, device_ids=dev_ids)

    logger.info("Building...")
    tp_model({"token_ids": np.zeros((1, 128), "int32"), "padding_mask": np.ones((1, 128), "int32")})
    

    logger.info("Compiling...")
    tp_model.compile(
        optimizer=keras.optimizers.SGD(LEARNING_RATE, momentum=0.0),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
        # --- FIX: Disable Eager, Enable JIT ---
        run_eagerly=False,  # Was True (Causes Host RAM crash)
        jit_compile=True    # Was False (Now safe due to train_step fix)
    )

    logger.info("Training...")
    tp_model.fit(train_ds, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH)
    logger.info("🎉 Success!")
    
    if hasattr(tp_model, 'temp_dir') and os.path.exists(tp_model.temp_dir):
        shutil.rmtree(tp_model.temp_dir)

if __name__ == "__main__":
    run_training()