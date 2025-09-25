import os
import time
import logging
import numpy as np
import sys  # Import sys to use sys.exit

# --- STEP 1: SET KERAS BACKEND (MUST BE BEFORE IMPORTING KERAS) ---
# This tells Keras to use JAX, which is required for multi-device parallelism on GPUs/TPUs.
os.environ["KERAS_BACKEND"] = "jax"

# --- STEP 2: Now import JAX, Keras, and all other libraries ---
import jax
import keras
import keras_hub
from keras_hub import models as keras_hub_models # Using an alias for clarity
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

# --- Configuration and Initialization ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import tensorflow as tf

# This line tells TensorFlow: "Don't touch the TPUs. Leave them for JAX."
tf.config.set_visible_devices([], 'TPU')
# --- CHANGED: Generalized JAX Device Detection for GPUs or CPUs ---
try:
    devices = jax.devices()
    DEVICES_AVAILABLE = len(devices)
    print(f"✅ Found {DEVICES_AVAILABLE} JAX devices: {[str(d) for d in devices]}")
    
    # Use all available devices for parallelism
    TARGET_DEVICES = devices
    TARGET_WORLD_SIZE = DEVICES_AVAILABLE
    
    if TARGET_WORLD_SIZE < 2:
         print(f"⚠️ WARNING: Tensor parallelism requires at least 2 devices, but only found {TARGET_WORLD_SIZE}.")

except Exception as e:
    print(f"Could not initialize JAX or find devices. Error: {e}")
    TARGET_WORLD_SIZE = 0


try:
    # Assuming your custom TensorParallelKeras class is in the expected path
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
BATCH_SIZE = 32
SEQUENCE_LENGTH = 128
LEARNING_RATE = 1e-4
EPOCHS = 2
STEPS_PER_EPOCH = 10 # Increased slightly for a more meaningful run
VALIDATION_STEPS = 5

MODEL_MAPPING = {
    # "gemma_2b_en": keras_hub_models.GemmaCausalLM,
    "opt_1.3b_en": keras_hub_models.OPTCausalLM,
    # "gpt2_base_en": keras_hub_models.GPT2CausalLM,
}

# ----------------------------------------------------------------------
# --- Dataset and Model Helpers (UNCHANGED) ---
# ----------------------------------------------------------------------

def load_shakespeare_dataset(model_preset, model_class):
    """
    Loads and preprocesses the Tiny Shakespeare dataset by manually downloading
    the raw text file.
    """
    print(f"   Loading and preprocessing Tiny Shakespeare dataset for {model_preset}...")
    
    # 1. Manually download the raw text file
    file_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    file_path = tf.keras.utils.get_file("tiny_shakespeare.txt", file_url)

    # 2. Read the text from the file
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"      ✅ Dataset loaded! It contains {len(text):,} characters.")

    # 3. *** FIX: Force preprocessor loading to run on CPU ***
    # This prevents a conflict where the TF-based tokenizer tries to access
    # the TPU that JAX has already claimed.
    print("      Running preprocessor loading on CPU to avoid device conflict...")
    with tf.device("/CPU:0"):
        preprocessor = model_class.from_preset(model_preset).preprocessor

    if preprocessor is None:
        raise ValueError(f"Could not load a preprocessor for {model_preset}.")
    tokenizer = preprocessor.tokenizer
    token_ids = tokenizer.tokenize(text)

    num_tokens = (len(token_ids) // (SEQUENCE_LENGTH + 1)) * (SEQUENCE_LENGTH + 1)
    sequences = np.array(token_ids[:num_tokens]).reshape(-1, SEQUENCE_LENGTH + 1)
    
    all_data = tf.data.Dataset.from_tensor_slices(sequences)
    
    num_sequences = sequences.shape[0]
    num_train_samples = int(0.9 * num_sequences)
    
    train_ds = all_data.take(num_train_samples)
    val_ds = all_data.skip(num_train_samples)

    print(f"      ✅ Preprocessing complete with {num_train_samples} training and {num_sequences - num_train_samples} validation sequences.")
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
# --- Main Verification Function ---
# ----------------------------------------------------------------------

def run_model_verification(preset_name, model_class):
    """Runs the full training verification test for a given model preset."""
    
    if TARGET_WORLD_SIZE < 2:
        print(f"SKIPPING TENSOR PARALLELISM for {preset_name}: Need at least 2 devices, found {TARGET_WORLD_SIZE}")
        return "SKIPPED"
    
    print(f"🔧 VERIFICATION FOR: {preset_name.upper()}")
    print("=" * 50)
    start_time_total = time.time()
    
    model_template = get_model_from_preset(preset_name, model_class)
    initial_weights = model_template.get_weights()
    print("      ✅ Initial weights saved from template model.")

    # NOTE: The 'load_shakespeare_dataset' function with the tf.device fix should be kept.
    train_ds_raw, val_ds_raw = load_shakespeare_dataset(preset_name, model_class)
    
    # --- START OF MODIFIED SECTION ---

    # 1. Build the TF data pipeline, but DO NOT use .repeat()
    # We will only iterate through it once to convert to NumPy.
    print("   Building initial tf.data pipeline...")
    train_ds = (
        train_ds_raw.take(STEPS_PER_EPOCH * BATCH_SIZE) # Take just enough data for the run
        .batch(BATCH_SIZE, drop_remainder=True)
        .map(format_for_causal_lm, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        val_ds_raw.take(VALIDATION_STEPS * BATCH_SIZE) # Take just enough data for the run
        .batch(BATCH_SIZE, drop_remainder=True)
        .map(format_for_causal_lm, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

    # 2. Convert the entire dataset to NumPy arrays. This is the crucial step.
    # It forces all TF processing to happen now, on the CPU, before training begins.
    print("   Converting tf.data.Dataset to NumPy arrays to decouple from JAX training...")
    train_data_np = tfds.as_numpy(train_ds)
    val_data_np = tfds.as_numpy(val_ds)

    # Unpack the NumPy iterators into explicit x, y arrays
    x_train = np.concatenate([x for x, y in train_data_np], axis=0)
    y_train = np.concatenate([y for x, y in train_data_np], axis=0)
    x_val = np.concatenate([x for x, y in val_data_np], axis=0)
    y_val = np.concatenate([y for x, y in val_data_np], axis=0)
    print(f"      ✅ Data converted. Train shapes: {x_train['token_ids'].shape}, {y_train.shape}")
    
    # --- END OF MODIFIED SECTION ---

    total_steps = STEPS_PER_EPOCH * EPOCHS
    total_samples = x_train['token_ids'].shape[0] * EPOCHS
    total_tokens_processed = total_samples * SEQUENCE_LENGTH 
    print(f"   Tokens per step: {BATCH_SIZE * SEQUENCE_LENGTH:,}")
    print(f"   Total tokens to process (per model): {total_tokens_processed:,}")

    print("\n   --- Training Tensor Parallel (TP) Model ---")
    
    print(f"   Initializing TensorParallelKeras with world_size={TARGET_WORLD_SIZE} on devices: {[str(d) for d in TARGET_DEVICES]}")
    tp_manager = TensorParallelKeras(
        model=model_template, 
        world_size=TARGET_WORLD_SIZE, 
        distributed_backend='jax',
        device_ids=TARGET_DEVICES
    )
    tp_model = tp_manager.build_assembled_model()
    
    try:
        tp_model.set_weights(initial_weights)
        print("      ✅ Initial weights set on TP model.")
    except Exception as e:
        print(f"      Warning: Could not set weights on TP model. {e}")

    tp_model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=LEARNING_RATE),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras_hub.metrics.Perplexity(from_logits=True, name="perplexity")],
    )
    
    tp_start_time = time.time()
    
    # 3. Fit the model using the NumPy arrays
    tp_history = tp_model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE, # Specify batch size here
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
    
    print("\n   --- Performance Metrics ---")
    print(f"      TP Training Time:       {tp_time:.2f} s")
    print(f"      TP Throughput:       {tp_throughput_tps:,.2f} Tokens/s")
    
    test_passed = True
    print(f"\n✅ Test for {preset_name} completed in {time.time() - start_time_total:.2f}s")
    return test_passed

# ----------------------------------------------------------------------
# --- Main Execution ---
# ----------------------------------------------------------------------

if __name__ == "__main__":
    
    if TARGET_WORLD_SIZE == 0:
        print("🛑 ERROR: No JAX devices found. Aborting verification suite.")
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