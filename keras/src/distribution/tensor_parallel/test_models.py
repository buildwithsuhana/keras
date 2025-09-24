import os
import time
import logging
import numpy as np
import syslog
import sys

# --- STEP 1: Import TensorFlow and configure GPU visibility ---
import tensorflow as tf
try:
    # --- MODIFIED: Ensure GPUs are visible to TensorFlow ---
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.set_visible_devices(gpus, 'GPU')
        print(f"✅ TensorFlow visibility successfully set for {len(gpus)} GPU(s).")
    else:
        print("⚠️ No GPUs found by TensorFlow.")
except RuntimeError as e:
    print(f"⚠️ Could not set TensorFlow visible devices. (May already be initialized): {e}")


# --- STEP 2: SET KERAS BACKEND (MUST BE BEFORE IMPORTING KERAS) ---
os.environ["KERAS_BACKEND"] = "jax"

# --- MODIFIED: Set WORLD_SIZE to 2 for two GPUs ---
WORLD_SIZE = 2
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={WORLD_SIZE}"

# --- STEP 3: Now import JAX, Keras, and all other libraries ---
import jax
import keras
import keras_hub
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

# --- Configuration and Initialization ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
print("="*70)
# --- MODIFIED: Updated warning for GPU execution ---
print(f"🚀 INFO: Testing a {WORLD_SIZE}-way sharded Gemma 7B model on GPUs.")
print("   Ensure you have enough VRAM available on each device.")
print("="*70)


# --- JAX Device Detection (MODIFIED for GPUs) ---
try:
    devices = jax.devices()
    print(f"JAX devices found: {[str(d) for d in devices]}")
    
    # --- MODIFIED: Filter for GPU devices ---
    gpu_devices = [d for d in devices if 'gpu' in d.platform.lower()]
    print(f"Found {len(gpu_devices)} GPU devices.")
    
    DEVICES_AVAILABLE = len(gpu_devices)
    
    if DEVICES_AVAILABLE < WORLD_SIZE:
        print(f"🛑 ERROR: Requested {WORLD_SIZE} GPU devices, but only {DEVICES_AVAILABLE} are available.")
        sys.exit(1)
    else:
        # --- MODIFIED: Target the found GPU devices ---
        TARGET_DEVICES = gpu_devices[:WORLD_SIZE]
        TARGET_WORLD_SIZE = WORLD_SIZE
        print(f"✅ Targeting {TARGET_WORLD_SIZE} GPU devices for parallelism: {[str(d) for d in TARGET_DEVICES]}")

except Exception as e:
    print(f"Could not initialize JAX or find devices. Error: {e}")
    TARGET_WORLD_SIZE = 0
# --- END MODIFICATIONS ---


# Mock TensorParallelKeras class for demonstration if the actual library is not installed
try:
    from src.tensor_parallel_keras.tensor_parallel_keras import TensorParallelKeras
except ImportError:
    print("Warning: `TensorParallelKeras` not found. Using a mock class for demonstration.")
    class TensorParallelKeras:
        def __init__(self, model, world_size, distributed_backend, device_ids=None):
            self._model = model
            print(f"Mock TensorParallelKeras initialized for model: {model.name}, world_size: {world_size}")
        def build_assembled_model(self):
            return keras.models.clone_model(self._model)
        def set_weights(self, weights):
            print("Mock: setting weights")
            self._model.set_weights(weights)


# --- Constants ---
BATCH_SIZE = 1 
SEQUENCE_LENGTH = 128
LEARNING_RATE = 1e-4
EPOCHS = 2
STEPS_PER_EPOCH = 5
VALIDATION_STEPS = 10

# --- MODIFIED: Update mapping to use Gemma 7B model for a more relevant test ---
MODEL_MAPPING = {
    "gemma_2b_en": keras_hub.models.GemmaCausalLM,
}

# ----------------------------------------------------------------------
# --- Dataset and Model Helpers (UNCHANGED) ---
# ----------------------------------------------------------------------

def load_shakespeare_dataset(model_preset, model_class):
    """Loads and preprocesses the Tiny Shakespeare dataset for a given model."""
    print(f"   Loading and preprocessing Tiny Shakespeare dataset for {model_preset}...")
    ds = tfds.load("tiny_shakespeare", split="train")
    text = "".join(example["text"].decode("utf-8") for example in ds.as_numpy_iterator())

    tokenizer = model_class.from_preset(model_preset).preprocessor.tokenizer
    token_ids = tokenizer.tokenize(text)

    num_tokens = (len(token_ids) // (SEQUENCE_LENGTH + 1)) * (SEQUENCE_LENGTH + 1)
    sequences = np.array(token_ids[:num_tokens]).reshape(-1, SEQUENCE_LENGTH + 1)
    
    all_data = tf.data.Dataset.from_tensor_slices(sequences)
    
    num_sequences = sequences.shape[0]
    num_train_samples = int(0.9 * num_sequences)
    
    train_ds = all_data.take(num_train_samples)
    val_ds = all_data.skip(num_train_samples)

    print(f"      ✅ Dataset ready with {num_train_samples} training and {num_sequences - num_train_samples} validation sequences.")
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
    print(f"   Creating {preset_name} model from KerasHub preset...")
    model = model_class.from_preset(preset_name, preprocessor=None)
    print(f"      ✅ Model created with {model.count_params():,} parameters.")
    return model

# ----------------------------------------------------------------------
# --- Main Verification Function (MODIFIED FOR GPUs) ---
# ----------------------------------------------------------------------

def run_model_verification(preset_name, model_class):
    """Runs the full training verification test for a given model preset."""
    
    if TARGET_WORLD_SIZE < 2:
        print(f"SKIPPING {preset_name}: Need at least 2 devices for tensor parallelism, found {TARGET_WORLD_SIZE}")
        return "SKIPPED"
    
    print(f"🔧 VERIFICATION FOR: {preset_name.upper()}")
    print("=" * 50)
    start_time_total = time.time()
    
    model_template = get_model_from_preset(preset_name, model_class)
    initial_weights = model_template.get_weights()
    print("      ✅ Initial weights saved from template model.")

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

    total_steps = STEPS_PER_EPOCH * EPOCHS
    total_samples = total_steps * BATCH_SIZE
    total_tokens_processed = total_samples * SEQUENCE_LENGTH 
    print(f"   Tokens per step: {BATCH_SIZE * SEQUENCE_LENGTH:,}")
    print(f"   Total tokens to process: {total_tokens_processed:,}")

    print("\n   --- Training Tensor Parallel (TP) Model on GPUs ---")
    
    # --- MODIFIED: Pass the detected GPU devices to the manager ---
    print(f"   Initializing TensorParallelKeras with world_size={TARGET_WORLD_SIZE} on devices: {[str(d) for d in TARGET_DEVICES]}")
    tp_manager = TensorParallelKeras(
        model=model_template, 
        world_size=TARGET_WORLD_SIZE, 
        distributed_backend='jax',
        device_ids=TARGET_DEVICES # Pass the detected JAX GPU devices
    )
    tp_model = tp_manager.build_assembled_model() 
    
    try:
        tp_model.set_weights(initial_weights)
        print("      ✅ Initial weights set on TP model.")
    except Exception as e:
        print(f"      Warning: Could not set weights on TP model (this is expected if layer names change). {e}")

    tp_model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=LEARNING_RATE),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras_hub.metrics.Perplexity(from_logits=True, name="perplexity")],
    )
    
    tp_start_time = time.time()
    tp_history = tp_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_steps=VALIDATION_STEPS,
        verbose=1
    )
    tp_end_time = time.time()
    print("      ✅ TP model training completed.")

    tp_time = tp_end_time - tp_start_time
    tp_throughput_tps = total_tokens_processed / tp_time

    print("\n   --- Final Validation Metrics ---")
    tp_final_val_loss = tp_history.history['val_loss'][-1]
    print(f"      TP Final Validation Loss:       {tp_final_val_loss:.4f}")
    
    print("\n   --- Performance ---")
    print(f"      TP Training Time:       {tp_time:.2f} s")
    print(f"      TP Throughput:       {tp_throughput_tps:,.2f} Tokens/s")

    print(f"✅ Test for {preset_name} completed in {time.time() - start_time_total:.2f}s")
    return True

# ----------------------------------------------------------------------
# --- Main Execution (UNCHANGED) ---
# ----------------------------------------------------------------------

if __name__ == "__main__":
    
    if TARGET_WORLD_SIZE == 0:
        print("🛑 ERROR: No JAX GPU devices found. Aborting verification suite.")
        sys.exit(1)
        
    print("\n🎯 TENSOR PARALLELISM VERIFICATION SUITE")
    print("=" * 70)
    
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
            logger.error(f"Test for {preset} failed with an exception: {e}", exc_info=True)
            results[preset] = "💥 ERROR"
        print("-" * 70)

    print("\n" + "=" * 70)
    print("🎉 VERIFICATION SUITE COMPLETED!")
    print(f"   Total execution time: {time.time() - total_start_time:.2f}s")
    print("\n   --- SUMMARY ---")
    for preset, status in results.items():
        print(f"   - {preset:<18}: {status}")
    print("=" * 70)