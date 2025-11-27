import logging
import os
import sys
import time
import gc

# --- OOM FIX 0: JAX Memory Configuration ---
# Disable eager preallocation to allow Keras to manage memory placement manually
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

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
    print(
        "Could not add project root to sys.path. "
        "Please run from the 'keras' directory or install as a package."
    )

# --- Backend and Device Configuration ---
os.environ["KERAS_BACKEND"] = "jax"
# Force host platform device count to simulate devices if needed (e.g. on CPU-only dev machines)
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

import jax
import keras_hub
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

import keras
# [FIX] Import config directly from src to avoid AttributeError
from keras.src.backend import config as backend_config
# [FIX] Import list_devices for correct TPU/GPU string IDs
from keras.src.distribution import list_devices 

# Hide GPUs from TensorFlow so it doesn't hog memory needed for JAX/Keras
tf.config.set_visible_devices([], "GPU")

# --- OOM FIX 1: Enable Mixed Precision / bfloat16 ---
backend_config.set_floatx("bfloat16")

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# --- Device Detection ---
try:
    # [FIX] Use Keras list_devices() instead of jax.devices() for compatibility
    devices = list_devices()
    logger.info(f"Devices found: {devices}")
    
    # Filter for accelerators (exclude CPUs if accelerators exist)
    accel_devices = [d for d in devices if "cpu" not in d.lower()]
    
    if accel_devices:
        TARGET_DEVICES = accel_devices
    else:
        TARGET_DEVICES = devices

    # [FIX] Dynamic world size based on available chips
    TARGET_WORLD_SIZE = len(TARGET_DEVICES)
        
    logger.info(
        f"Targeting {TARGET_WORLD_SIZE} devices for parallelism: {TARGET_DEVICES}"
    )

except Exception as e:
    logger.error(f"Could not initialize JAX or find devices. Error: {e}")
    TARGET_WORLD_SIZE = 0
    TARGET_DEVICES = []


from keras.src.distribution.tensor_parallel.tensor_parallel_keras import (
    TensorParallelKeras,
)

# --- Constants ---
BATCH_SIZE = 1  # Increased slightly as LoRA saves memory
SEQUENCE_LENGTH = 32
LEARNING_RATE = 1e-4 # Adjusted for Adafactor
EPOCHS = 1
STEPS_PER_EPOCH = 10
VALIDATION_STEPS = 5

MODEL_MAPPING = {
    "gemma2_9b_en": keras_hub.models.GemmaCausalLM, 
}

# ----------------------------------------------------------------------
# --- Dataset and Model Helpers ---
# ----------------------------------------------------------------------

def load_shakespeare_dataset(model_preset, model_class):
    """Loads and preprocesses the Tiny Shakespeare dataset."""
    logger.info(
        f"Loading and preprocessing Tiny Shakespeare dataset for {model_preset}..."
    )
    ds = tfds.load("tiny_shakespeare", split="train", as_supervised=False)
    text = "".join(
        example["text"].decode("utf-8") for example in ds.as_numpy_iterator()
    )

    logger.info("Loading tokenizer (on CPU)...")
    with keras.device("cpu"):
        # Use the specific tokenizer class if available to avoid loading model weights
        try:
            tokenizer = keras_hub.models.GemmaTokenizer.from_preset(model_preset)
        except:
            temp_model = model_class.from_preset(model_preset)
            tokenizer = temp_model.preprocessor.tokenizer
            del temp_model
        gc.collect()

    token_ids = tokenizer.tokenize(text)

    num_tokens = (len(token_ids) // (SEQUENCE_LENGTH + 1)) * (
        SEQUENCE_LENGTH + 1
    )
    sequences = np.array(token_ids[:num_tokens]).reshape(
        -1, SEQUENCE_LENGTH + 1
    )

    all_data = tf.data.Dataset.from_tensor_slices(sequences)

    num_sequences = sequences.shape[0]
    num_train_samples = int(0.9 * num_sequences)

    train_ds = all_data.take(num_train_samples)
    val_ds = all_data.skip(num_train_samples)

    logger.info(
        f"Dataset ready with {num_train_samples} training and "
        f"{num_sequences - num_train_samples} validation sequences."
    )
    return train_ds, val_ds


def format_for_causal_lm(data):
    features = {
        "token_ids": data[:, :-1],
        "padding_mask": tf.ones_like(data[:, :-1], dtype=tf.bool),
    }
    labels = data[:, 1:]
    return features, labels


def get_model_from_preset(preset_name, model_class):
    """Creates a CausalLM model from a KerasNLP preset."""
    logger.info(f"Creating {preset_name} model from KerasNLP preset (on CPU)...")
    
    with keras.device("cpu"):
        model = model_class.from_preset(preset_name, preprocessor=None)
        
        # [FIX] Enable LoRA to fit optimizer states in memory
        if "gemma" in preset_name:
            logger.info("✨ Enabling LoRA (Rank=4) for memory efficiency...")
            model.backbone.enable_lora(rank=4)
        
    logger.info(f"Model created. Trainable params: {model.count_params():,}")
    return model


# ----------------------------------------------------------------------
# --- Plotting Function ---
# ----------------------------------------------------------------------

def plot_training_graphs(tp_history, preset_name):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle(f"{preset_name} - Tensor Parallel Training Metrics", fontsize=16)

    ax1.plot(
        tp_history.history["val_loss"],
        label="Validation Loss",
        color="green",
        marker="o",
    )
    ax1.set_title("Validation Loss")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(
        tp_history.history["val_perplexity"],
        label="Validation Perplexity",
        color="purple",
        marker="o",
    )
    ax2.set_title("Validation Perplexity")
    ax2.set_ylabel("Perplexity")
    ax2.legend()
    ax2.grid(True)

    output_filename = f"{preset_name}_tp_training.png"
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_filename)
    logger.info(f"\nTraining graph saved to {output_filename}")
    plt.close()


# ----------------------------------------------------------------------
# --- Main Verification Function ---
# ----------------------------------------------------------------------

def run_model_verification(preset_name, model_class):
    if TARGET_WORLD_SIZE < 2:
        logger.warning(f"SKIPPING {preset_name}: Need at least 2 devices.")
        return "SKIPPED"

    logger.info(f"--- VERIFICATION FOR: {preset_name.upper()} ---")

    train_ds_raw, val_ds_raw = load_shakespeare_dataset(preset_name, model_class)

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

    logger.info("\n--- Training Tensor Parallel (TP) Model ---")
    
    tp_model_template = get_model_from_preset(preset_name, model_class)

    tp_model = TensorParallelKeras(
        model=tp_model_template,
        device_count=TARGET_WORLD_SIZE,
        device_ids=TARGET_DEVICES,
    )

    # [FIX] Use Adafactor instead of AdamW to save memory on T4/TPU
    tp_model.compile(
        optimizer=keras.optimizers.Adafactor(learning_rate=LEARNING_RATE),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras_hub.metrics.Perplexity(from_logits=True, name="perplexity")
        ],
    )

    tp_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_steps=VALIDATION_STEPS,
        verbose=1,
    )
    
    logger.info("TP model training completed successfully.")
    
    plot_training_graphs(tp_model.history, preset_name)
    
    del tp_model
    del tp_model_template
    keras.backend.clear_session()
    gc.collect()

    return True


# ----------------------------------------------------------------------
# --- Main Execution ---
# ----------------------------------------------------------------------

if __name__ == "__main__":
    if TARGET_WORLD_SIZE == 0:
        logger.critical("No JAX devices found. Aborting.")
        sys.exit(1)

    logger.info("\n" + "=" * 70)
    logger.info("      TENSOR PARALLELISM EXECUTION SUITE")
    logger.info("=" * 70)
    logger.info(f"Global Dtype: {backend_config.floatx()}")

    results = {}
    total_start_time = time.time()

    for preset, model_class in MODEL_MAPPING.items():
        try:
            result = run_model_verification(preset, model_class)
            if result == "SKIPPED":
                results[preset] = "⚪ SKIPPED"
            else:
                results[preset] = "✅ PASS" if result else "❌ FAIL"
        except Exception as e:
            logger.error(
                f"Test for {preset} failed with an exception: {e}",
                exc_info=True,
            )
            results[preset] = "💥 ERROR"
        logger.info("-" * 70)

    logger.info("\n" + "=" * 70)
    logger.info("🎉 EXECUTION SUITE COMPLETED!")
    logger.info(
        f"   Total execution time: {time.time() - total_start_time:.2f}s"
    )
    logger.info("\n   --- SUMMARY ---")
    for preset, status in results.items():
        print(f"   - {preset:<18}: {status}")
    logger.info("=" * 70)