#!/usr/bin/env python3
"""Test script to verify the fix for the None shape TypeError in distributed training."""

import os
# Set torch backend before any other keras imports
os.environ["KERAS_BACKEND"] = "torch"
os.environ["KERAS_DISTRIBUTION_DEBUG"] = "1"

import keras
from keras import layers
from keras.src.distribution import DataParallel, list_devices
import numpy as np

def test_optimizer_creation_with_distribution():
    """Test that optimizer creation works within a distribution scope."""
    print("Testing optimizer creation within DataParallel scope...")
    
    # Get devices
    devices = list_devices("gpu")
    if not devices:
        devices = ["cpu:0"]
    
    print(f"Using devices: {devices}")
    
    # Create DataParallel distribution
    dp = DataParallel(devices=devices, auto_shard_dataset=False)
    print(f"✓ DataParallel created with mesh_shape={dp.device_mesh.shape}")
    
    # Create model and optimizer within distribution scope
    with dp.scope():
        model = keras.Sequential([
            layers.Dense(128, activation="relu", input_shape=(64,)),
            layers.Dense(64, activation="relu"),
            layers.Dense(10)
        ])
        
        total_params = model.count_params()
        print(f"✓ Model created with {total_params:,} parameters")
        
        # This should now work without TypeError
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        print(f"✓ Optimizer created: {optimizer.name}")
        
        model.compile(optimizer=optimizer, loss="mse")
    
    # Quick training test
    batch_size = 32
    x = np.random.random((batch_size, 64)).astype("float32")
    y = np.random.random((batch_size, 10)).astype("float32")
    
    print("Running one training step...")
    with dp.scope():
        history = model.fit(x, y, epochs=1, verbose=0)
        loss = history.history['loss'][0]
    
    print(f"✓ Training step completed. Loss: {loss:.6f}")
    print("\n✓ All tests PASSED!")
    return True

if __name__ == "__main__":
    try:
        test_optimizer_creation_with_distribution()
    except Exception as e:
        print(f"\n✗ Test FAILED with error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

