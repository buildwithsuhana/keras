#!/usr/bin/env python3
"""
Test script to verify the DTensor mixed tensor fix.

This tests that the convert_to_tensor function properly converts
regular torch.Tensors to DTensors when a device mesh is active.
"""

import os
# Must be set before any other imports
os.environ["KERAS_BACKEND"] = "torch"

import torch
from torch.distributed._tensor import DTensor, DeviceMesh, Replicate

# Import the fixed function
import sys
sys.path.insert(0, '/Users/suhanaaa/keras')

from keras.src.backend.torch.core import convert_to_tensor
from keras.src.backend.torch.distribution_lib import (
    _to_backend_mesh,
    _get_default_device_mesh
)
from keras.src.distribution import DeviceMesh as KerasDeviceMesh


def test_convert_to_tensor_with_mesh():
    """Test that convert_to_tensor converts tensors to DTensors when mesh is active."""
    print("Test 1: convert_to_tensor with active device mesh")
    
    # Create a Keras DeviceMesh
    devices = ["cuda:0", "cuda:1"] if torch.cuda.is_available() else ["cpu"]
    keras_mesh = KerasDeviceMesh(
        shape=(len(devices),),
        axis_names=["model"],
        devices=devices
    )
    
    # Convert to PyTorch DeviceMesh and set as global
    torch_mesh = _to_backend_mesh(keras_mesh)
    
    # Create a regular torch tensor
    if torch.cuda.is_available():
        tensor = torch.randn(8, 64, device="cuda")
    else:
        tensor = torch.randn(8, 64)
    
    print(f"  Input tensor type: {type(tensor)}")
    
    # Call convert_to_tensor - it should convert to DTensor
    result = convert_to_tensor(tensor)
    
    print(f"  Output tensor type: {type(result)}")
    
    # Verify it's a DTensor
    assert isinstance(result, DTensor), \
        f"Expected DTensor, got {type(result)}"
    
    print("  ✓ Test PASSED: convert_to_tensor correctly converts to DTensor\n")
    
    return True


def test_numpy_conversion():
    """Test that numpy arrays are converted to DTensors when mesh is active."""
    print("Test 2: convert_to_tensor with numpy array and active device mesh")
    
    import numpy as np
    
    # Create a Keras DeviceMesh
    devices = ["cuda:0", "cuda:1"] if torch.cuda.is_available() else ["cpu"]
    keras_mesh = KerasDeviceMesh(
        shape=(len(devices),),
        axis_names=["model"],
        devices=devices
    )
    
    # Convert to PyTorch DeviceMesh and set as global
    torch_mesh = _to_backend_mesh(keras_mesh)
    
    # Create a numpy array
    arr = np.random.random((8, 64)).astype("float32")
    
    print(f"  Input type: {type(arr)}")
    
    # Call convert_to_tensor
    result = convert_to_tensor(arr)
    
    print(f"  Output tensor type: {type(result)}")
    
    # Verify it's a DTensor
    assert isinstance(result, DTensor), \
        f"Expected DTensor, got {type(result)}"
    
    print("  ✓ Test PASSED: numpy arrays are correctly converted to DTensors\n")
    
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("DTensor Mixed Tensor Fix Verification")
    print("=" * 60)
    print()
    
    try:
        test_convert_to_tensor_with_mesh()
        test_numpy_conversion()
        
        print("=" * 60)
        print("ALL TESTS PASSED!")
        print("The fix correctly converts tensors to DTensors when a mesh is active.")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

