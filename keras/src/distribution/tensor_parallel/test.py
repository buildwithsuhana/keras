import os

# --- 1. Environment Setup (Must be before importing Keras) ---
os.environ["KERAS_BACKEND"] = "jax"

# Simulate 8 devices on CPU for testing logic (Remove this on real TPU)
# This creates a "fake" mesh of 8 devices.
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import keras
import numpy as np
import jax
import keras.src.applications as keras_applications
# --- Import your custom modules ---
# Adjust these imports based on your actual folder structure
from keras.src.distribution.distribution_lib import list_devices
from keras.src.distribution.distribution_lib import DeviceMesh
from keras.src.distribution.distribution_lib import AutoTPDistribution
from keras.src.distribution.tensor_parallel.lazy_init import lazy_init_scope

def print_section(title):
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def main():
    print_section("1. Device Setup")
    
    # Verify JAX sees the devices
    devices = list_devices()
    print(f"Total Devices Detected: {len(devices)}")
    print(f"Device List: {devices[:4]} ...")

    if len(devices) < 8:
        print("⚠️ Warning: Expected 8 devices for this test mesh (2x4).")
        # Adjusting mesh to fit available devices if necessary
        mesh_shape = (1, len(devices))
        axis_names = ('data', 'model')
    else:
        mesh_shape = (2, 4) # 2 Data Parallel, 4 Tensor Parallel
        axis_names = ('data', 'model')

    device_mesh = DeviceMesh(shape=mesh_shape, axis_names=axis_names, devices=devices[:8])
    print(f"Device Mesh Configured: {mesh_shape} {axis_names}")

    # --- 2. Lazy Initialization ---
    print_section("2. Lazy Model Instantiation")
    
    # We use a standard ResNet here, but this works identical for Gemma 9B
    # The key is that 'model' variable here will consume almost 0 RAM.
    from keras.src.applications.resnet import ResNet50

    print("Defining model metadata (Lazy Init)...")
    with lazy_init_scope():
        # Use the imported class directly
        model = ResNet50(weights=None, classes=1000)

    print(f"Model Object Created: {model.name}")
    
    # Verification: Check a random weight to ensure it's a LazyVariable
    first_layer_kernel = model.layers[2].kernel # Conv1_conv
    print(f"Inspection of 'conv1_conv' kernel: {first_layer_kernel}")
    
    if "LazyVariable" not in str(type(first_layer_kernel)) and "LazyVariable" not in str(first_layer_kernel):
        raise RuntimeError("❌ Lazy Initialization failed! Weights are real tensors.")
    else:
        print("✅ Confirmed: Weights are LazyVariables (Ghosts).")

    # --- 3. Auto Sharding Distribution ---
    print_section("3. Applying AutoTPDistribution")
    
    # This step triggers the analysis of the LazyVariables.
    # It will calculate split rules based on shapes.
    # It will then materialize ONLY the shards on the devices.
    try:
        distribution = AutoTPDistribution(model, device_mesh=device_mesh)
    except Exception as e:
        print(f"❌ Crash during distribution creation: {e}")
        raise e

    print("✅ Distribution Wrapper Created.")
    print(f"   Internal Shards: {len(distribution.model.model_shards)}")

    # Verify parameters are sharded
    # Let's look at the same layer in the first shard
    shard_0 = distribution.model.model_shards[0]
    
    # We need to find the specific kernel we looked at earlier
    # Note: internal names might differ slightly due to flattening, 
    # but let's check the parameter count.
    
    print("\n--- Verifying Memory Usage (Sharding) ---")
    
    # Calculate expected size vs actual size for a specific Dense layer
    # ResNet "predictions" layer is Dense(1000). Input is 2048.
    # Full size: 2048 * 1000 = 2,048,000 parameters.
    # Split across 4 devices (Column Parallel): 2048 * 250 = 512,000 parameters.
    
    found_layer = False
    for weight in shard_0.weights:
        if "predictions" in weight.name and "kernel" in weight.name:
            found_layer = True
            shape = weight.shape
            print(f"Layer: {weight.name}")
            print(f"   Original Global Shape: (2048, 1000)")
            print(f"   Actual Shard 0 Shape:  {shape}")
            
            # Check logic (FIXED)
            # Scenario A: Column Parallel (Split dim 1)
            if shape[1] < 1000:
                 print(f"   ✅ SUCCESS: Column Sliced (1000 -> {shape[1]})")
            
            # Scenario B: Row Parallel (Split dim 0) - This is what happened!
            elif shape[0] < 2048:
                 print(f"   ✅ SUCCESS: Row Sliced (2048 -> {shape[0]})")
            
            # Scenario C: Replicated
            else:
                 print("   ⚠️ WARNING: Weight is NOT sliced (Replicated)")
            
    if not found_layer:
        print("⚠️ Could not find 'predictions' layer to verify sharding.")

    # --- 4. Compile and Train Step ---
    print_section("4. Compilation & Execution")

    with distribution.scope():
        distribution.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        print("✅ Model Compiled with Coordinated Optimizer")

        # Create dummy data
        batch_size = 16
        # Input shape for ResNet50 is (224, 224, 3)
        x_train = np.random.random((batch_size, 224, 224, 3)).astype("float32")
        y_train = np.random.randint(0, 1000, (batch_size, 1)).astype("float32")
        
        print("🚀 Starting fit() for 1 epoch...")
        distribution.model.fit(x_train, y_train, epochs=1, batch_size=batch_size)
        print("✅ Training step completed successfully.")

if __name__ == "__main__":
    main()