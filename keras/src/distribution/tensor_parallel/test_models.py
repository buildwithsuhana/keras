import logging
import os
import sys
import time

import numpy as np
# ---------------------------------------------------------------------------
# MUST BE SET BEFORE IMPORTING TENSORFLOW, JAX, OR KERAS
# ---------------------------------------------------------------------------
# 1. Disable Splash Attention (Fixes ConcretizationTypeError)
os.environ["KERAS_BACKEND_CONFIG_SKIP_SPLASH_ATTENTION"] = "1"
os.environ["GEMMA_DISABLE_PALLAS"] = "true"

# 2. Minimize Memory Fragmentation (Helps with the 81MB OOM error)
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90" # Leave 10% for runtime overhead
# --- DEBUG MODE: ON ---
# Uncomment this line to force JAX to run line-by-line.
# This will show you exactly which line causes the "ConcretizationTypeError".
# WARNING: This will be slow and might OOM if the model is massive, 
# but it is the best way to find logic bugs.

# os.environ["JAX_DISABLE_JIT"] = "1"
# ---------------------------------------------------------------------------
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
# Force JAX to see "CPU" devices as separate XLA devices for testing if no GPUs
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

import jax
import keras_hub
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

import keras
keras.mixed_precision.set_global_policy("mixed_bfloat16")

# Hide TF GPUs so JAX takes full control
tf.config.set_visible_devices([], "GPU")

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# --- JAX Device Detection ---
try:
    devices = jax.devices()
    logger.info(f"JAX devices found: {[str(d) for d in devices]}")
    host_devices = [d for d in devices if d.platform == "cpu"]
    if not host_devices:
        host_devices = devices

    DEVICES_AVAILABLE = len(host_devices)
    WORLD_SIZE = 4 # Set this to 4 or 8 if testing large models on multi-gpu

    if DEVICES_AVAILABLE < WORLD_SIZE:
        logger.warning(
            f"Requested {WORLD_SIZE} devices, but only {DEVICES_AVAILABLE} available."
        )
        TARGET_DEVICES = host_devices
        TARGET_WORLD_SIZE = DEVICES_AVAILABLE
    else:
        TARGET_DEVICES = host_devices[:WORLD_SIZE]
        TARGET_WORLD_SIZE = WORLD_SIZE
        logger.info(
            f"Targeting the first {TARGET_WORLD_SIZE} devices for parallelism: "
            f"{[str(d) for d in TARGET_DEVICES]}"
        )

except Exception as e:
    logger.error(f"Could not initialize JAX or find devices. Error: {e}")
    TARGET_WORLD_SIZE = 0


from keras.src.distribution.tensor_parallel.tensor_parallel_keras import (
    TensorParallelKeras,
)

# --- Constants ---
BATCH_SIZE = 1
SEQUENCE_LENGTH = 128
LEARNING_RATE = 1e-4
EPOCHS = 2
STEPS_PER_EPOCH = 5
VALIDATION_STEPS = 5

MODEL_MAPPING = {
    # "opt_6.7b_en": keras_hub.models.OPTCausalLM,
    # You can now add "gemma_7b_en" here without crashing OOM!
    "gemma2_2b_en": keras_hub.models.GemmaCausalLM, 
}

# ----------------------------------------------------------------------
# --- Dataset and Model Helpers ---
# ----------------------------------------------------------------------

def load_shakespeare_dataset(model_preset):
    """Loads and preprocesses the Tiny Shakespeare dataset."""
    logger.info(
        f"Loading and preprocessing Tiny Shakespeare dataset for {model_preset}..."
    )
    ds = tfds.load("tiny_shakespeare", split="train", as_supervised=False)
    text = "".join(
        example["text"].decode("utf-8") for example in ds.as_numpy_iterator()
    )

    # Use a separate tokenizer instance just for data prep
    tokenizer = keras_hub.models.GemmaTokenizer.from_preset(model_preset)
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
# --- Main Verification Function (MODIFIED FOR OOM SAFETY) ---
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

    # --- Memory Logger Helper ---
    def log_memory_status(step_name):
        import gc
        gc.collect()
        keras.backend.clear_session()
        try:
            import psutil
            vm_mem = psutil.virtual_memory()
            logger.info(f"[{step_name}] Host RAM: {vm_mem.percent}% used")
        except:
            pass

    # --- 1. Tensor Parallel Model Initialization (LAZY) ---
    logger.info("\n--- Training Tensor Parallel (TP) Model ---")
    
    def model_builder():
        return model_class.from_preset(
            preset_name, 
            preprocessor=None, 
            load_weights=False
        )

    log_memory_status("Before Init")

    # Correct Initialization
    tp_model = TensorParallelKeras(
        model_input=model_builder,
        device_count=TARGET_WORLD_SIZE,
        device_ids=TARGET_DEVICES,
    )

    log_memory_status("After Init")

    # Correct Compilation
    tp_model.compile(
        optimizer=keras.optimizers.Adafactor(
            learning_rate=LEARNING_RATE,
            weight_decay=0.01
        ),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras_hub.metrics.Perplexity(from_logits=True, name="perplexity")
        ],
    )

    log_memory_status("After Compile")

    # --- 2. Training ---
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

    # --- 3. Verification ---
    logger.info("\n--- ⚖️ Verification Results ---")
    logger.info(f"TP Final Validation Loss: {tp_final_val_loss:.6f}")

    plot_training_graphs(tp_history, preset_name)

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
    logger.info("      TENSOR PARALLELISM EXECUTION SUITE (OOM-SAFE)")
    logger.info("=" * 70)

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