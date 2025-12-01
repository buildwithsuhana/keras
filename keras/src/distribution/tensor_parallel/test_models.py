import os
import gc
import shutil
import logging
import time
import numpy as np

# --- 1. Environment Setup ---
# CRITICAL: Do NOT set "XLA_FLAGS" for host device count if you are on GPU.
# It forces JAX to use CPU!
os.environ["KERAS_BACKEND"] = "jax"

# Optional: Aggressive memory freeing
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import jax
import jax.nn
import jax.numpy as jnp
import keras
import keras_hub
import tensorflow as tf
import tensorflow_datasets as tfds

# Strict bfloat16 policy for efficiency
keras.config.set_dtype_policy("bfloat16")

# Hide GPUs from TensorFlow so JAX gets exclusive access
tf.config.set_visible_devices([], "GPU")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- 🩹 ADVANCED MONKEY PATCH (Fixes GQA + Driver Mismatch) ---
try:
    import keras.src.backend.jax.nn as jax_keras_nn
    
    def safe_dot_product_attention(query, key, value, bias=None, mask=None, scale=None, is_causal=False, **kwargs):
        q_heads = query.shape[-2]
        k_heads = key.shape[-2]
        
        if q_heads != k_heads:
            rep_factor = q_heads // k_heads
            key = jnp.repeat(key, rep_factor, axis=-2)
            value = jnp.repeat(value, rep_factor, axis=-2)

        return jax.nn.dot_product_attention(
            query, key, value, 
            bias=bias, 
            mask=None, 
            scale=scale, 
            is_causal=is_causal
        )
        
    jax_keras_nn.dot_product_attention = safe_dot_product_attention
    logger.info("🩹 Patch Applied: Enabled Manual GQA.")
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

# --- 🕵️ DEBUGGER HELPERS ---
def print_memory_stats(stage_name):
    """Prints the memory usage to detect leaks/OOMs."""
    try:
        # Check the first device (usually gpu:0 or tpu:0)
        device = jax.local_devices()[0]
        stats = device.memory_stats()
        gb_in_use = stats.get('bytes_in_use', 0) / 1e9
        gb_limit = stats.get('bytes_limit', 0) / 1e9
        utilization = (gb_in_use / gb_limit) * 100 if gb_limit > 0 else 0
        
        logger.info(f"📊 MEMORY [{device.platform}:{device.id}] @ {stage_name}")
        logger.info(f"   ↳ Used: {gb_in_use:.2f} GB / {gb_limit:.2f} GB ({utilization:.1f}%)")
    except Exception:
        pass

def inspect_model_shards(tp_model, target_devices):
    """Verifies that shards are actually living on different devices."""
    logger.info("🕵️ INSPECTING SHARD PLACEMENT...")
    
    for i, shard in enumerate(tp_model.model_shards):
        expected_device = target_devices[i]
        
        if not shard.trainable_variables:
            logger.warning(f"   ⚠️ Shard {i} has no trainable variables!")
            continue
            
        first_var = shard.trainable_variables[0]
        
        # Robustly get device info
        try:
            # Check where the underlying JAX array lives
            if hasattr(first_var, 'value') and hasattr(first_var.value, 'devices'):
                devs = list(first_var.value.devices())
                actual_device = str(devs[0]) if devs else "Unknown"
            elif hasattr(first_var, 'device'):
                actual_device = str(first_var.device)
            else:
                actual_device = "Unknown"
        except Exception as e:
            actual_device = f"Error: {e}"

        # Loose matching because 'gpu:0' != 'cuda:0' strictly
        match = str(expected_device).lower().replace("cuda", "gpu") in actual_device.lower().replace("cuda", "gpu")
        status = "✅ OK" if match else "❌ WRONG DEVICE"
        logger.info(f"   Shard {i} | Expect: {expected_device} | Actual: {actual_device} | {status}")

def inspect_optimizer_state(tp_model):
    """Checks where the optimizer variables are being allocated."""
    logger.info("🕵️ INSPECTING OPTIMIZER STATE...")
    
    if not hasattr(tp_model.optimizer, 'variables'):
        return

    vars = tp_model.optimizer.variables
    if not vars:
        logger.info("   Optimizer has 0 variables.")
        return

    on_device0 = 0
    on_cpu = 0
    for v in vars:
        try:
            d = "unknown"
            if hasattr(v, 'value') and hasattr(v.value, 'devices'):
                d = str(list(v.value.devices())[0]).lower()
            elif hasattr(v, 'device'):
                d = str(v.device).lower()
            
            if "gpu:0" in d or "tpu:0" in d: on_device0 += 1
            if "cpu" in d: on_cpu += 1
        except:
            pass
        
    logger.info(f"   Optimizer Variables Total: {len(vars)}")
    logger.info(f"   ↳ On Device:0 : {on_device0} (⚠️ Should be low)")
    logger.info(f"   ↳ On CPU      : {on_cpu} (✅ Preferred for Coordinator)")

# ---------------------------------------------------------------

def get_devices():
    devices = jax.devices()
    logger.info(f"JAX Devices: {devices}")
    # Filter for TPUs or GPUs
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
    
    total_tokens = (len(tokens) // (SEQUENCE_LENGTH + 1)) * (SEQUENCE_LENGTH + 1)
    tokens = tokens[:total_tokens].reshape(-1, SEQUENCE_LENGTH + 1)
    
    dataset = tf.data.Dataset.from_tensor_slices(tokens)
    
    def prepare_batch(batch):
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
    with keras.device("cpu"):
        model = keras_hub.models.GemmaCausalLM.from_preset(MODEL_PRESET)
        return model

def run_training():
    print_memory_stats("STARTUP")
    
    device_count, target_devices = get_devices()
    logger.info(f"Accelerator Devices detected: {device_count}")
    
    if device_count < 2:
        logger.error("Need at least 2 accelerators for Tensor Parallelism.")
        return

    gc.collect()
    jax.clear_caches()

    train_ds = load_data(MODEL_PRESET)

    logger.info("Preparing Tensor Parallel Model...")
    # Clean string conversion for devices
    device_ids = [f"{d.platform}:{d.id}" for d in target_devices]
    
    tp_model = TensorParallelKeras(
        model=model_factory, 
        device_count=device_count,
        device_ids=device_ids
    )
    print_memory_stats("AFTER TP INIT")

    logger.info("🔧 Manually building model...")
    dummy_inputs = {
        "token_ids": np.zeros((BATCH_SIZE, SEQUENCE_LENGTH), dtype="int32"),
        "padding_mask": np.ones((BATCH_SIZE, SEQUENCE_LENGTH), dtype="int32"),
    }
    tp_model(dummy_inputs)
    print_memory_stats("AFTER BUILD")
    
    inspect_model_shards(tp_model, target_devices)

    logger.info("Compiling model with SGD...")
    optimizer = keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=0.0)

    tp_model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
        jit_compile=False
    )
    
    try:
        tp_model.optimizer.build(tp_model.trainable_variables)
        inspect_optimizer_state(tp_model)
    except Exception as e:
        logger.warning(f"Could not manually build optimizer: {e}")

    logger.info("Starting Training Loop...")
    try:
        tp_model.fit(train_ds, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH)
        logger.info("🎉 Success!")
    except Exception as e:
        logger.error(f"❌ Training Failed: {e}")
        print_memory_stats("CRASH STATE")
        raise e
    finally:
        if hasattr(tp_model, 'temp_dir') and os.path.exists(tp_model.temp_dir):
            shutil.rmtree(tp_model.temp_dir)

if __name__ == "__main__":
    run_training()