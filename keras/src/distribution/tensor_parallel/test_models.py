import logging
import os
import sys
import time

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
    print("Could not add project root to sys.path.")

# --- Backend Configuration ---
os.environ["KERAS_BACKEND"] = "jax"
# TPU v5e usually doesn't need force_host_platform, but we keep it just in case of sim
# os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4" 

import jax
import keras_hub
import tensorflow as tf
import tensorflow_datasets as tfds
import keras

# --- 1. MEMORY OPTIMIZATION: Mixed Precision ---
# Bfloat16 is native to TPU and uses half the memory of float32
keras.config.set_dtype_policy("mixed_bfloat16")

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- JAX Device Detection (TPU Logic) ---
try:
    devices = jax.devices()
    logger.info(f"JAX devices found: {[str(d) for d in devices]}")
    
    tpu_devices = [d for d in devices if d.platform == "tpu"]
    
    if tpu_devices:
        TARGET_DEVICES = tpu_devices
        logger.info(f"✅ Using {len(TARGET_DEVICES)} TPU Cores (v5e detected).")
    else:
        TARGET_DEVICES = devices
        logger.warning(f"⚠️ No TPUs found! Using {devices[0].platform}.")

    TARGET_WORLD_SIZE = len(TARGET_DEVICES)

except Exception as e:
    logger.error(f"Could not initialize JAX. Error: {e}")
    sys.exit(1)

from keras.src.distribution.distribution_lib import DeviceMesh
from keras.src.distribution.distribution_lib import AutoTPDistribution

# --- Constants ---
# Kept strictly low for memory safety check
BATCH_SIZE = 1 
SEQUENCE_LENGTH = 128  # Increased slightly from 8 to be realistic, but still small
LEARNING_RATE = 1e-4   # Slightly lower for Adafactor
EPOCHS = 1
STEPS_PER_EPOCH = 5
VALIDATION_STEPS = 2

# --- Model Selection ---
MODEL_PRESET = "gemma2_9b_en"
MODEL_CLASS = keras_hub.models.GemmaCausalLM

# ----------------------------------------------------------------------
# --- Dataset Helpers ---
# ----------------------------------------------------------------------

def load_shakespeare_dataset(model_preset):
    """Loads and preprocesses the Tiny Shakespeare dataset."""
    logger.info(f"Loading dataset for {model_preset}...")
    ds = tfds.load("tiny_shakespeare", split="train", as_supervised=False)
    text = "".join(example["text"].decode("utf-8") for example in ds.as_numpy_iterator())

    # Tokenize on CPU to save TPU memory
    with keras.device("cpu"):
        # Auto-select correct tokenizer (GemmaTokenizer)
        tokenizer = keras_hub.models.GemmaTokenizer.from_preset(model_preset)
        token_ids = tokenizer.tokenize(text)

    num_tokens = (len(token_ids) // (SEQUENCE_LENGTH + 1)) * (SEQUENCE_LENGTH + 1)
    sequences = np.array(token_ids[:num_tokens]).reshape(-1, SEQUENCE_LENGTH + 1)
    all_data = tf.data.Dataset.from_tensor_slices(sequences)
    
    num_train = int(0.9 * sequences.shape[0])
    train_ds = all_data.take(num_train)
    val_ds = all_data.skip(num_train)
    return train_ds, val_ds

def format_for_causal_lm(data):
    features = {
        "token_ids": data[:, :-1],
        "padding_mask": tf.ones_like(data[:, :-1], dtype=tf.bool),
    }
    labels = data[:, 1:]
    return features, labels

# --- MODEL BUILDER ---
def model_builder_factory(preset_name, model_class):
    def model_builder(**kwargs):
        logger.info(f"Creating {preset_name} model (Zero-Stage Init)...")
        model = model_class.from_preset(preset_name, preprocessor=None, **kwargs)
        
        # --- MEMORY OPTIMIZATION: LoRA ---
        # Rank 4 is sufficient for verification and uses minimal memory.
        logger.info("--- Enabling LoRA (Rank=4) ---") 
        if hasattr(model, "enable_lora"):
             model.enable_lora(rank=4)
        elif hasattr(model, "backbone") and hasattr(model.backbone, "enable_lora"):
             model.backbone.enable_lora(rank=4)
        
        # Log parameter counts
        total_params = model.count_params()
        trainable_params = sum(np.prod(w.shape) for w in model.trainable_variables)
        logger.info(f"Model created. Total: {total_params:,} | Trainable: {trainable_params:,}")
        
        return model
    return model_builder

# ----------------------------------------------------------------------
# --- Physical Sharding Verification ---
# ----------------------------------------------------------------------
def inspect_physical_memory(model):
    logger.info("\n" + "="*50)
    logger.info("🔍 PERFORMING PHYSICAL MEMORY INSPECTION")
    logger.info("="*50)
    
    target_var = None
    # Gemma specific search: look for 'gating_dense' (FFN) or 'query_dense' (Attn)
    for v in model.trainable_variables:
        if "gating_dense" in v.path and "kernel" in v.path:
            target_var = v
            break
            
    if target_var is None:
        target_var = model.trainable_variables[0]

    jax_array = target_var.value
    
    try:
        # Check size of the shard on the FIRST chip
        local_shard = jax_array.addressable_shards[0].data
        
        global_shape = jax_array.shape
        local_shape = local_shard.shape
        
        global_params = np.prod(global_shape)
        local_params = np.prod(local_shape)
        
        logger.info(f"   Target Variable:        {target_var.path}")
        logger.info(f"   Logical Shape (Global): {global_shape} ({global_params:,} params)")
        logger.info(f"   Physical Shape (Chip):  {local_shape}  ({local_params:,} params)")
        
        if local_params < global_params:
            ratio = global_params / local_params
            logger.info(f"   🎉 SUCCESS: Variable is physically sharded!")
            logger.info(f"   ⬇️  Memory footprint reduced by {ratio:.1f}x per chip.")
        else:
            logger.warning("   ⚠️ WARNING: Variable appears REPLICATED.")
    except Exception as e:
        logger.error(f"   Could not inspect addressable shards. Error: {e}")
    
    logger.info("="*50 + "\n")


# ----------------------------------------------------------------------
# --- Main Execution ---
# ----------------------------------------------------------------------

def run_model_verification(preset_name, model_class):
    if TARGET_WORLD_SIZE < 2:
        logger.warning("⚠️ Running on < 2 devices. Tensor Parallelism is disabled.")

    logger.info(f"--- RUNNING: {preset_name} on {TARGET_WORLD_SIZE} TPU CORES ---")

    # Data Pipeline
    train_ds_raw, val_ds_raw = load_shakespeare_dataset(preset_name)
    train_ds = (
        train_ds_raw.batch(BATCH_SIZE, drop_remainder=True)
        .map(format_for_causal_lm, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
        .repeat()
    )
    val_ds = (
        val_ds_raw.batch(BATCH_SIZE, drop_remainder=True)
        .map(format_for_causal_lm, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
        .repeat()
    )

    # 1. Initialize AutoTPDistribution
    device_mesh = DeviceMesh(
        shape=(1, TARGET_WORLD_SIZE), 
        axis_names=('data', 'model'), 
        devices=TARGET_DEVICES
    )

    # 2. Prepare Builder
    model_builder_fn = model_builder_factory(preset_name, model_class)

    # 3. Initialize Strategy
    distribution = AutoTPDistribution(
        model_builder_fn, 
        device_mesh=device_mesh,
    )
    
    logger.info("\n--- Initializing & Sharding on TPU ---")

    # 4. Build & Distribute
    # FIX: No 'with distribution.scope()' here. The class handles it lazily.
    tp_model = distribution.model 

    # 5. VERIFY SHARDING
    inspect_physical_memory(tp_model)

    # 6. Compile & Train
    # --- MEMORY OPTIMIZATION: Adafactor ---
    # Adafactor uses significantly less memory than AdamW (no momentum buffers).
    logger.info("🚀 Compiling with Adafactor (Low Memory Optimizer)...")
    optimizer = keras.optimizers.Adafactor(learning_rate=LEARNING_RATE)
    
    tp_model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras_hub.metrics.Perplexity(from_logits=True, name="perplexity")
        ],
    )

    logger.info("🏃 Starting training loop...")
    tp_history = tp_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_steps=VALIDATION_STEPS,
        verbose=1,
    )
    
    logger.info("✅ SUCCESS: TPU Training Complete")
    return True

if __name__ == "__main__":
    try:
        run_model_verification(MODEL_PRESET, MODEL_CLASS)
    except Exception as e:
        logger.exception("TPU Run Failed")