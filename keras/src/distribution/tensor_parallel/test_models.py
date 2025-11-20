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
    print(
        "Could not add project root to sys.path. "
        "Please run from the 'keras' directory or install as a package."
    )

# --- Backend and Device Configuration ---
os.environ["KERAS_BACKEND"] = "jax"
# JAX Memory Management (Prevents pre-allocation crashes)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".95"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import jax
import keras_hub
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

import keras

# --- 1. CRITICAL: Enable Mixed Precision ---
# Computation in float16/bfloat16, Storage in int8 (QLoRA)
keras.config.set_dtype_policy("mixed_float16") 

# Ensure TF doesn't hog GPU memory needed by JAX
tf.config.set_visible_devices([], "GPU")

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# --- JAX Device Detection ---
try:
    devices = jax.devices()
    logger.info(f"JAX devices found: {[str(d) for d in devices]}")
    
    # Prefer GPUs
    gpu_devices = [d for d in devices if d.platform == "gpu"]
    if gpu_devices:
        TARGET_DEVICES = gpu_devices
        logger.info(f"✅ Using {len(TARGET_DEVICES)} GPUs.")
    else:
        # Fallback to CPU
        TARGET_DEVICES = [d for d in devices if d.platform == "cpu"]
        logger.warning("⚠️ No GPUs found! Using CPU.")

    DEVICES_AVAILABLE = len(devices)
    TARGET_WORLD_SIZE = len(TARGET_DEVICES)
    # You can override WORLD_SIZE here if you want to limit GPUs used
    WORLD_SIZE = TARGET_WORLD_SIZE 

except Exception as e:
    logger.error(f"Could not initialize JAX or find devices. Error: {e}")
    TARGET_WORLD_SIZE = 0


from keras.src.distribution import DeviceMesh
from keras.src.distribution import AutoTPDistribution

# --- Constants ---
BATCH_SIZE = 4         # Start small (4 or 8)
SEQUENCE_LENGTH = 64   # Reduced from 128 to save VRAM
LEARNING_RATE = 1e-4
EPOCHS = 1
STEPS_PER_EPOCH = 10
VALIDATION_STEPS = 5

MODEL_MAPPING = {
    "gemma_7b_en": keras_hub.models.GemmaCausalLM,
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

    # Use CPU for tokenization to save GPU memory
    with keras.device("cpu"):
        tokenizer = keras_hub.models.GemmaTokenizer.from_preset(
            model_preset
        )
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


# --- MODIFIED MODEL BUILDER (QLoRA Fix) ---
def model_builder_factory(preset_name, model_class):
    """Returns a callable function that builds the model, required for OOM safety."""
    
    def model_builder(**kwargs):
        logger.info(f"Creating {preset_name} model (inside scope)...")
        
        # --- 2. CRITICAL: Use Int8 Quantization ---
        # This enables 8-bit weight loading (QLoRA style), dropping static
        # memory usage by ~50%. Essential for 7B models on T4 GPUs.
        kwargs["dtype"] = "int8" 
        
        # Create Model (Sharded automatically by AutoTPDistribution scope)
        model = model_class.from_preset(preset_name, preprocessor=None, **kwargs)
        
        # --- 3. CRITICAL: Enable LoRA ---
        # Reduces optimizer memory from 26GB to ~100MB.
        logger.info("--- Enabling LoRA (Rank=4) ---")
        
        if hasattr(model, "enable_lora"):
             model.enable_lora(rank=4)
        elif hasattr(model, "backbone") and hasattr(model.backbone, "enable_lora"):
             logger.info("Called enable_lora on model.backbone")
             model.backbone.enable_lora(rank=4)
        else:
             logger.warning("⚠️ enable_lora not found! Training full weights (High OOM Risk).")
        
        total_params = model.count_params()
        # Safe count for sharded/quantized variables
        trainable_params = sum(np.prod(w.shape) for w in model.trainable_variables)
        
        logger.info(f"Model created. Total params: {total_params:,}")
        logger.info(f"Trainable params (LoRA only): {trainable_params:,}")
        
        return model
        
    return model_builder


# ----------------------------------------------------------------------
# --- Plotting Function ---
# ----------------------------------------------------------------------

def plot_training_graphs(tp_history, preset_name):
    """Plots and saves the loss and perplexity graphs for the TP run."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle(f"{preset_name} - Tensor Parallel Training Metrics", fontsize=16)

    if "val_loss" in tp_history.history:
        ax1.plot(tp_history.history["val_loss"], label="Validation Loss", color="green", marker="o")
    ax1.set_title("Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax1.grid(True)

    if "val_perplexity" in tp_history.history:
        ax2.plot(tp_history.history["val_perplexity"], label="Validation Perplexity", color="purple", marker="o")
    ax2.set_title("Validation Perplexity")
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
    tensor parallelism with the OOM-safe workflow.
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

    logger.info("\n--- Setting up AutoTPDistribution for OOM-Safe Build ---")
    
    # Create mesh for GPUs
    device_mesh = DeviceMesh(
        shape=(1, TARGET_WORLD_SIZE), 
        axis_names=('data', 'model'), 
        devices=TARGET_DEVICES
    )

    # Prepare the builder
    model_builder_fn = model_builder_factory(preset_name, model_class)

    # Initialize Strategy
    # Note: We don't pass dtype="float16" here because we handle it explicitly 
    # inside the model_builder_factory as "int8" for weights.
    distribution = AutoTPDistribution(
        model_builder_fn, 
        device_mesh=device_mesh
    )
    
    logger.info("\n--- Training Tensor Parallel (TP) Model (Inside Safe Scope) ---")

    # Trigger safe build
    with distribution.scope():
        tp_model = distribution.model 

    # Compile with AdamW
    tp_model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=LEARNING_RATE),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras_hub.metrics.Perplexity(from_logits=True, name="perplexity")
        ],
    )

    # Fit
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

    # --- Verification ---
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
    logger.info("      TENSOR PARALLELISM EXECUTION SUITE")
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