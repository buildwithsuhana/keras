import os
# Limit JAX to 90% of VRAM to prevent fragmentation OOMs
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"

# --- 1. Memory Tuning ---
# Disable preallocation to allow memory to grow as needed
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["KERAS_BACKEND"] = "jax"
# Ensure CPU host device count is sufficient for JAX
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import logging
import sys
import gc
import shutil
import numpy as np
import jax
import keras
import keras_hub
import tensorflow as tf
import tensorflow_datasets as tfds

# --- 2. Mixed Precision is Mandatory ---
# bfloat16 is crucial for 9B models on T4 GPUs
keras.config.set_dtype_policy("bfloat16")

tf.config.set_visible_devices([], "GPU")

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from keras.src.distribution.tensor_parallel.tensor_parallel_keras import (
    TensorParallelKeras,
)

MODEL_PRESET = "gemma2_9b_en"
# With LoRA + TP on 2xT4s, we can slightly increase batch size
BATCH_SIZE = 1
SEQUENCE_LENGTH = 128
LEARNING_RATE = 5e-5
EPOCHS = 1
STEPS_PER_EPOCH = 5

def get_devices():
    try:
        devices = jax.devices()
        logger.info(f"Available JAX devices: {[str(d) for d in devices]}")
        accel_devices = [d for d in devices if d.platform != "cpu"]
        if len(accel_devices) >= 2:
            return 2, accel_devices[:2]
        else:
            logger.warning("Not enough GPUs found. Using CPU devices.")
            cpu_devices = [d for d in devices if d.platform == "cpu"]
            return 2, cpu_devices[:2]
    except Exception as e:
        logger.error(f"Device detection failed: {e}")
        return 0, []

def load_data(preset):
    logger.info("Loading Tiny Shakespeare dataset...")
    ds = tfds.load("tiny_shakespeare", split="train", as_supervised=False)
    text = "".join(ex["text"].decode("utf-8") for ex in ds.as_numpy_iterator())
    
    logger.info("Tokenizing...")
    tokenizer = keras_hub.models.GemmaTokenizer.from_preset(preset)
    tokens = tokenizer(text[:20000]) 
    if isinstance(tokens, dict): tokens = tokens["token_ids"]
    tokens = np.array(tokens)
    
    total_tokens = (len(tokens) // (SEQUENCE_LENGTH + 1)) * (SEQUENCE_LENGTH + 1)
    tokens = tokens[:total_tokens].reshape(-1, SEQUENCE_LENGTH + 1)
    
    dataset = tf.data.Dataset.from_tensor_slices(tokens)
    def prepare_batch(batch):
        return ({"token_ids": batch[:-1], "padding_mask": tf.ones_like(batch[:-1], dtype=tf.bool)}, batch[1:])

    return dataset.batch(BATCH_SIZE, drop_remainder=True).map(prepare_batch, num_parallel_calls=tf.data.AUTOTUNE)

def run_training():
    device_count, target_devices = get_devices()
    if device_count < 2:
        logger.error("Aborting: Need at least 2 devices.")
        return

    train_ds = load_data(MODEL_PRESET)

    logger.info("Preparing Tensor Parallel Model...")
    
    # --- 3. LoRA Factory (The Fix) ---
    # We define the model creation logic here so the Master Model (18GB)
    # is loaded, offloaded to disk, and then deleted from RAM immediately.
    from keras.src.layers.preprocessing.image_preprocessing import transform
    # Import RematScope
    from keras import RematScope

    def model_factory():
        logger.info(f"🏭 Factory: Loading {MODEL_PRESET}...")
        
        # WRAP model creation in RematScope
        # mode="activations" is usually sufficient and safer/faster than "full"
        with RematScope(mode="activations"):
            with keras.device("cpu"):
                model = keras_hub.models.GemmaCausalLM.from_preset(MODEL_PRESET)
                
                logger.info("✨ Enabling LoRA (Rank=4)...")
                model.backbone.enable_lora(rank=4)
                return model

    # Pass the FACTORY (callable), not the MODEL
    tp_model = TensorParallelKeras(
        model=model_factory, 
        device_count=device_count,
        device_ids=[str(d) for d in target_devices]
    )
    
    # At this point, the Master Model is guaranteed to be deleted from RAM.
    
    logger.info("Compiling model...")
    # Use SGD instead of AdamW to save optimizer state memory
    # Use gradient_accumulation_steps to simulate a batch size of 16 (1 * 16)
    optimizer = keras.optimizers.SGD(
        learning_rate=LEARNING_RATE, 
        momentum=0.9,
        gradient_accumulation_steps=16 
    )

    tp_model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )

    logger.info("Starting Training Loop...")
    tp_model.fit(train_ds, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH)
    logger.info("🎉 Training Finished Successfully!")

    # Cleanup temp dirs if any remained (optional)
    if hasattr(tp_model, 'temp_dir') and os.path.exists(tp_model.temp_dir):
        shutil.rmtree(tp_model.temp_dir)

if __name__ == "__main__":
    run_training()