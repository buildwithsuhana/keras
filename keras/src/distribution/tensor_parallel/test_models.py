import os

# 1. Configuration: Set backend to JAX
os.environ["KERAS_BACKEND"] = "jax"
# Optional: XLA Flags
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

# 2. Fix for TensorFlow-JAX conflict (Must happen before other imports)
import tensorflow as tf
try:
    # Prevent TF from grabbing GPU memory so JAX can use it
    tf.config.set_visible_devices([], 'GPU') 
    tf.config.set_visible_devices([], 'TPU')
except Exception as e:
    # Ignore if TF is not fully installed or devices not found
    pass

import jax
import numpy as np
import keras

# 3. Import KerasNLP (now safe)
import keras_nlp 

# Import your local distribution utilities
from keras.src.distribution.distribution_lib import lazy_init_scope
from keras.src.distribution.tensor_parallel.tensor_parallel_keras import TensorParallelKeras
from keras.src.distribution import list_devices

def run_training():
    print("🚀 Starting Zero-Stage Tensor Parallel Training for Gemma 9B")

    # --- Step 1: Device Detection ---
    devices = list_devices()
    device_count = len(devices)
    print(f"✅ Detected {device_count} devices: {devices}")
    
    if device_count < 2:
        print("⚠️ Warning: Tensor Parallelism typically requires 2+ devices.")

    # --- Step 2: Lazy Initialization (The "Zero Stage") ---
    print("\n--- Phase 1: Lazy Initialization ---")
    print("\n--- Phase 1: Lazy Initialization ---")
    print("Defining Gemma 9B model structure without allocating weights...")
    
    with lazy_init_scope():
        # Load model structure only (no weights)
        # CHANGED: "gemma_9b_en" -> "gemma2_9b_en"
        model = keras_nlp.models.GemmaCausalLM.from_preset(
            "gemma2_9b_en",  # <--- CORRECT PRESET NAME
            load_weights=False 
        )
    
    print("✅ Lazy model created.")

    # --- Step 3: Materialization & Sharding ---
    print("\n--- Phase 2: Sharding & Materialization ---")
    print(f"Distributing model across {device_count} devices...")
    
    # This allocates real memory directly on devices
    tp_model = TensorParallelKeras(
        model, 
        device_count=device_count
    )

    print("✅ Model successfully sharded.")
    
    # --- Step 4: Training Setup ---
    print("\n--- Phase 3: Compilation & Fitting ---")
    
    keras.mixed_precision.set_global_policy("mixed_bfloat16")

    tp_model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=5e-5),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )

    # Dummy Data
    print("Generating dummy training data...")
    batch_size = 8
    seq_length = 128
    
    x_train = np.random.randint(0, 256000, size=(64, seq_length))
    y_train = np.random.randint(0, 256000, size=(64, seq_length))

    print("Starting fit...")
    history = tp_model.fit(
        x_train, 
        y_train, 
        batch_size=batch_size, 
        epochs=1
    )
    
    print("✅ Training complete.")
    return history

if __name__ == "__main__":
    run_training()