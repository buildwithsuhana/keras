#!/usr/bin/env python3
"""Test script to verify DataParallel fix for DTensor gradient tracking."""

import os
# MUST be set before any other imports
os.environ["KERAS_BACKEND"] = "torch"
os.environ["KERAS_DISTRIBUTION_DEBUG"] = "1"

import torch
import torch.distributed as dist
import keras
from keras import layers
from keras.src.distribution import DataParallel, list_devices
from keras.src.backend.torch import distribution_lib
import numpy as np


def test_device_mesh_registration():
    """Test that DeviceMesh is properly registered in global state."""
    print("\n" + "=" * 60)
    print("TEST 1: DeviceMesh Registration")
    print("=" * 60)
    
    devices = list_devices("gpu")
    if not devices:
        devices = ["cpu:0"]
    
    print(f"Using devices: {devices}")
    
    # Create DataParallel distribution
    dp = DataParallel(devices=devices, auto_shard_dataset=False)
    print(f"DataParallel created with mesh shape: {dp.device_mesh.shape}")
    
    # Check if backend mesh is registered
    backend_mesh = distribution_lib._get_default_device_mesh()
    print(f"Backend mesh from global state: {backend_mesh}")
    
    if backend_mesh is not None:
        print(f"✓ Backend mesh registered: shape={backend_mesh.shape}, dim_names={backend_mesh.mesh_dim_names}")
    else:
        print("✗ Backend mesh NOT registered")
    
    return backend_mesh is not None


def test_distribute_variable():
    """Test distribute_variable function with registered mesh."""
    print("\n" + "=" * 60)
    print("TEST 2: distribute_variable with Registered Mesh")
    print("=" * 60)
    
    devices = list_devices("gpu")
    if not devices:
        devices = ["cpu:0"]
    
    dp = DataParallel(devices=devices, auto_shard_dataset=False)
    
    # Create a test tensor
    test_tensor = np.random.random((64, 128)).astype("float32")
    
    # Test with replicated layout (all None)
    layout = (None, None)
    print(f"Testing distribute_variable with layout={layout}")
    
    result = distribution_lib.distribute_variable(test_tensor, layout)
    print(f"Result type: {type(result)}")
    print(f"Result requires_grad: {result.requires_grad}")
    
    if isinstance(result, torch.nn.Parameter):
        print(f"✓ Created Parameter with requires_grad={result.requires_grad}")
        return True
    else:
        print(f"✗ Expected torch.nn.Parameter, got {type(result)}")
        return False


def test_gradient_flow():
    """Test gradient flow with DataParallel."""
    print("\n" + "=" * 60)
    print("TEST 3: Gradient Flow")
    print("=" * 60)
    
    devices = list_devices("gpu")
    if not devices:
        devices = ["cpu:0"]
    
    dp = DataParallel(devices=devices, auto_shard_dataset=False)
    
    with dp.scope():
        model = keras.Sequential([
            layers.Dense(128, activation="relu", input_shape=(64,)),
            layers.Dense(64, activation="relu"),
            layers.Dense(10)
        ])
        
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss="mse")
    
    # Create training data
    x = np.random.random((32, 64)).astype("float32")
    y = np.random.random((32, 10)).astype("float32")
    
    # Training step
    with dp.scope():
        history = model.fit(x, y, epochs=1, verbose=0)
    
    print(f"✓ Training completed with loss: {history.history['loss'][0]:.6f}")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("DataParallel DTensor Fix Verification")
    print("=" * 60)
    
    # Check PyTorch and DTensor availability
    print(f"PyTorch version: {torch.__version__}")
    print(f"DTensor available: {distribution_lib.DTENSOR_AVAILABLE}")
    
    results = []
    
    # Run tests
    results.append(("DeviceMesh Registration", test_device_mesh_registration()))
    results.append(("distribute_variable", test_distribute_variable()))
    results.append(("Gradient Flow", test_gradient_flow()))
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_passed = True
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED - DataParallel fix is working!")
    else:
        print("SOME TESTS FAILED - Check the output above")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

