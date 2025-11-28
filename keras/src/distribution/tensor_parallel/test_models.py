import os

# 1. Configuration: Set backend to JAX (preferred for TPU)
os.environ["KERAS_BACKEND"] = "jax"

# Optional: XLA Flags for TPU optimization
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8" # Simulating 8 devices if no TPU present

import jax
import numpy as np
import keras
import keras_nlp  # Requires keras-nlp for Gemma

# Import the modified distribution utilities
# Ensure your PYTHONPATH includes the path to your modified Keras source
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
    print("Defining Gemma 9B model structure without allocating weights...")
    
    # The context manager ensures Layer.add_weight returns LazyVariables (placeholders)
    # No OOM will occur here, even for 9B+ parameter models on a single CPU.
    with lazy_init_scope():
        # Instantiate the model. 
        # Note: We use 'from_preset' which normally loads weights. 
        # Inside the scope, the loading is deferred/skipped or handled lazily.
        # If loading pre-trained weights, ensure load_weights logic respects the lazy scope
        # or load them *after* sharding. For 'from_preset' with structure only:
        model = keras_nlp.models.GemmaCausalLM.from_preset(
            "gemma_9b_en",
            load_weights=False # Important: Define structure first, load weights later if needed
        )
    
    print("✅ Lazy model created. Checking variable types...")
    # Verify that the variables are indeed LazyVariables
    sample_var = model.layers[2].weights[0]
    print(f"   Sample weight type: {type(sample_var)}") 
    # Expected: <class 'keras.src.distribution.distribution_lib.LazyVariable'>

    # --- Step 3: Materialization & Sharding ---
    print("\n--- Phase 2: Sharding & Materialization ---")
    print(f"Distributing model across {device_count} devices...")
    
    # This step does the heavy lifting:
    # 1. It iterates through the LazyModel.
    # 2. It calculates the slice required for each device (Tensor Parallelism).
    # 3. It allocates *only* that slice on the specific device.
    # 4. It destroys the LazyVariable metadata as it goes.
    tp_model = TensorParallelKeras(
        model, 
        device_count=device_count
    )

    print("✅ Model successfully sharded.")
    
    # --- Step 4: Verification ---
    print("\n--- Phase 3: Verification ---")
    # We check the first shard to see if it holds real data on device
    first_shard = tp_model.model_shards[0]
    first_var = first_shard.trainable_variables[0]
    
    print(f"   Shard 0 Variable Device: {first_var.device}")
    print(f"   Shard 0 Variable Shape:  {first_var.shape}")
    
    # Confirm the original lazy model is cleaned up (optional check)
    if tp_model._original_model is None:
        print("   Original full model has been garbage collected (Memory Freed).")

    # --- Step 5: Training ---
    print("\n--- Phase 4: Compilation & Fitting ---")
    
    # Enable mixed precision for v5e performance
    keras.mixed_precision.set_global_policy("mixed_bfloat16")

    tp_model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=5e-5),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )

    # Dummy Data for demonstration
    print("Generating dummy training data...")
    batch_size = 8  # Global batch size
    seq_length = 128
    
    # Create dummy inputs (integers for tokens)
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