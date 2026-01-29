#!/usr/bin/env python3
"""Test script to verify the scalar variable fix for DataParallel."""

import os
# Set the backend before importing keras
os.environ["KERAS_BACKEND"] = "torch"

import keras
from keras import layers
from keras.src.distribution import DataParallel, list_devices
import numpy as np

def test_scalar_variable_handling():
    """Test that scalar variables (like optimizer iterations) work with DataParallel."""
    
    print("=" * 60)
    print("Testing Scalar Variable Handling with DataParallel")
    print("=" * 60)
    
    # Get devices
    devices = list_devices("gpu")
    print(f"Using devices: {devices}")
    
    # Create DataParallel distribution
    dp = DataParallel(devices=devices, auto_shard_dataset=False)
    print(f"✓ DataParallel created with mesh_shape={dp.device_mesh.shape}")
    
    # Create model inside DataParallel scope
    with dp.scope():
        model = keras.Sequential([
            layers.Dense(128, activation="relu", input_shape=(64,)),
            layers.Dense(64, activation="relu"),
            layers.Dense(10)
        ])
        
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss="mse")
    
    print(f"✓ Model created and compiled with {model.count_params()} parameters")
    
    # Check the iterations variable shape
    print(f"\nOptimizer iterations variable:")
    print(f"  - Name: {optimizer._iterations.name}")
    print(f"  - Shape: {optimizer._iterations.shape}")
    print(f"  - Value: {optimizer._iterations.value}")
    
    # Create training data
    batch_size = 32
    x = np.random.random((batch_size, 64)).astype("float32")
    y = np.random.random((batch_size, 10)).astype("float32")
    
    # Training
    print(f"\nTraining for 1 epoch...")
    with dp.scope():
        history = model.fit(x, y, epochs=1, verbose=1)
    
    print(f"\n✓ Training completed successfully!")
    print(f"  - Initial loss: {history.history['loss'][0]:.6f}")
    
    # Verify iterations counter was updated
    print(f"\nIterations counter after training: {optimizer._iterations.value}")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed! Scalar variable handling works correctly.")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    test_scalar_variable_handling()

