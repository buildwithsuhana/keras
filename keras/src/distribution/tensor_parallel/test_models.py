import logging
import os
import sys
import time
import gc

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

# Hide GPUs from TensorFlow so it doesn't hog memory needed for JAX/Keras
tf.config.set_visible_devices([], "GPU")

# --- OOM FIX 1: Enable Mixed Precision / bfloat16 ---
# This drastically reduces memory usage on both Host (CPU) and Device (TPU/GPU).
# bfloat16 is the native format for TPU v5e.
# FIX: Use set_floatx instead of set_dtype
keras.config.set_floatx("bfloat16")

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# --- JAX Device Detection ---
try:
    devices = jax.devices()
    logger.info(f"JAX devices found: {[str(d) for d in devices]}")
    host_devices = [d for d in devices if d.platform == "cpu"]
    
    # If we are on a TPU VM, 'devices' will usually be the TPUs.
    # If on a CPU machine, we use the CPUs.
    if len(devices) > 1 and devices[0].platform != "cpu":
        TARGET_DEVICES = devices
    else:
        # Fallback for local testing or CPU simulation
        TARGET_DEVICES = host_devices

    DEVICES_AVAILABLE = len(TARGET_DEVICES)
    # Set this to the number of chips you want to shard across (e.g., 4 or 8 for TPU v5e)
    WORLD_SIZE = 4 

    if DEVICES_AVAILABLE < WORLD_SIZE:
        logger.warning(
            f"Requested {WORLD_SIZE} devices, but only {DEVICES_AVAILABLE} available."
        )
        TARGET_WORLD_SIZE = DEVICES_AVAILABLE
        # Adjust target devices to what we actually have
        TARGET_DEVICES = TARGET_DEVICES[:TARGET_WORLD_SIZE]
    else:
        TARGET_DEVICES = TARGET_DEVICES[:WORLD_SIZE]
        TARGET_WORLD_SIZE = WORLD_SIZE
        
    logger.info(
        f"Targeting {TARGET_WORLD_SIZE} devices for parallelism: "
        f"{[str(d) for d in TARGET_DEVICES]}"
    )

except Exception as e:
    logger.error(f"Could not initialize JAX or find devices. Error: {e}")
    TARGET_WORLD_SIZE = 0
    TARGET_DEVICES = []


from keras.src.distribution.tensor_parallel.tensor_parallel_keras import (
    TensorParallelKeras,
)

# --- Constants ---
BATCH_SIZE = 1 # Reduce this if you still hit OOM
SEQUENCE_LENGTH = 32
LEARNING_RATE = 1e-4
EPOCHS = 1
STEPS_PER_EPOCH = 5
VALIDATION_STEPS = 5

MODEL_MAPPING = {
    # "opt_125m_en": keras_hub.models.OPTCausalLM,
    # Uncomment these to test larger models
    # "gemma2_2b_en": keras_hub.models.GemmaCausalLM,
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

    # --- OOM FIX 2: Load Tokenizer Safely ---
    # We load the tokenizer inside a CPU scope. 
    # Ideally, we should use the Preprocessor class directly to avoid loading weights,
    # but using the model class with CPU scope is a safe generic fallback.
    logger.info("Loading tokenizer (on CPU)...")
    with keras.device("cpu"):
        # We assume the model class has a 'preprocessor' property or we instantiate it
        # temporarily just to grab the tokenizer. 
        # Note: This might still load weights into RAM, but 'cpu' ensures they don't hit TPU.
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
    """Formats data for KerasNLP's CausalLM, creating features and labels."""
    features = {
        "token_ids": data[:, :-1],
        "padding_mask": tf.ones_like(data[:, :-1], dtype=tf.bool),
    }
    labels = data[:, 1:]
    return features, labels


def get_model_from_preset(preset_name, model_class):
    """Creates a CausalLM model from a KerasNLP preset."""
    logger.info(f"Creating {preset_name} model from KerasNLP preset (on CPU)...")
    
    # --- OOM FIX 3: Initialize Original Model on CPU ---
    # This ensures the full 9B/125M parameters are allocated in Host RAM, not HBM.
    # TensorParallelKeras will then slice these and only move the slices to HBM.
    with keras.device("cpu"):
        model = model_class.from_preset(preset_name, preprocessor=None)
        
    logger.info(f"Model created with {model.count_params():,} parameters.")
    return model


# ----------------------------------------------------------------------
# --- Plotting Function ---
# ----------------------------------------------------------------------

def plot_training_graphs(tp_history, preset_name):
    """Plots and saves the loss and perplexity graphs for the TP run."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle(f"{preset_name} - Tensor Parallel Training Metrics", fontsize=16)

    # Plotting Loss
    ax1.plot(
        tp_history.history["val_loss"],
        label="Tensor Parallel - Validation Loss",
        color="green",
        linestyle="-",
        marker="o",
    )
    ax1.set_title("Validation Loss")
    ax1.set_ylabel("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax1.grid(True)

    # Plotting Perplexity
    ax2.plot(
        tp_history.history["val_perplexity"],
        label="Tensor Parallel - Validation Perplexity",
        color="purple",
        linestyle="-",
        marker="o",
    )
    ax2.set_title("Validation Perplexity")
    ax2.set_ylabel("Perplexity")
    ax2.set_xlabel("Epoch")
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
    """
    Runs a training execution test for a given model preset using
    tensor parallelism.
    """
    if TARGET_WORLD_SIZE < 2:
        logger.warning(
            f"SKIPPING {preset_name}: Need at least 2 devices for tensor "
            f"parallelism, found {TARGET_WORLD_SIZE}"
        )
        return "SKIPPED"

    logger.info(f"--- VERIFICATION FOR: {preset_name.upper()} ---")

    # --- Common Setup ---
    # Pass model_class to helper to safely load tokenizer
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

    # --- 1. Tensor Parallel Model Training ---
    logger.info("\n--- Training Tensor Parallel (TP) Model ---")
    
    # Helper now enforces CPU placement
    tp_model_template = get_model_from_preset(preset_name, model_class)

    tp_model = TensorParallelKeras(
        model=tp_model_template,
        device_count=TARGET_WORLD_SIZE,
        device_ids=TARGET_DEVICES,
    )

    tp_model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=LEARNING_RATE),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras_hub.metrics.Perplexity(from_logits=True, name="perplexity")
        ],
    )

    tp_history = tp_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_steps=VALIDATION_STEPS,
        verbose=1,
    )
    tp_final_val_loss = tp_history.history["val_loss"][-1]
    logger.info("TP model training completed successfully.")

    # --- 2. Verification ---
    logger.info("\n--- ⚖️ Verification Results ---")
    logger.info(f"TP Final Validation Loss: {tp_final_val_loss:.6f}")

    plot_training_graphs(tp_history, preset_name)
    
    # Cleanup to free memory for next model in loop
    del tp_model
    del tp_model_template
    keras.backend.clear_session()
    gc.collect()

    logger.info("✅ SUCCESS: TP model training finished without errors.")
    return True


# ----------------------------------------------------------------------
# --- Main Execution ---
# ----------------------------------------------------------------------

if __name__ == "__main__":
    if TARGET_WORLD_SIZE == 0:
        logger.critical("No JAX devices found. Aborting verification suite.")
        sys.exit(1)

    logger.info("\n" + "=" * 70)
    logger.info("      TENSOR PARALLELISM EXECUTION SUITE")
    logger.info("=" * 70)
    logger.info(f"Global Dtype: {keras.config.dtype_policy().name}")

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