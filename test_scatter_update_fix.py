#!/usr/bin/env python3
"""Test script to verify scatter_update fix for DTensor support."""

import os
# Must be set before any other imports
os.environ["KERAS_BACKEND"] = "torch"

import torch
import numpy as np

# Test 1: Basic scatter_update with regular tensors (original behavior)
print("=" * 60)
print("TEST 1: Basic scatter_update with regular tensors")
print("=" * 60)

from keras.src.backend.torch.core import scatter_update

inputs = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32)
indices = np.array([[0, 1]], dtype=np.int64)
updates = np.array([[10, 20]], dtype=np.float32)

result = scatter_update(inputs, indices, updates)
expected = np.array([[10, 20, 3, 4], [5, 6, 7, 8]], dtype=np.float32)

print(f"Input: {inputs}")
print(f"Indices: {indices}")
print(f"Updates: {updates}")
print(f"Result: {result}")
print(f"Expected: {expected}")
assert np.allclose(result, expected), f"Test 1 failed: {result} != {expected}"
print("✓ Test 1 PASSED: Basic scatter_update works\n")


# Test 2: scatter_update with DTensor (the fix we made)
print("=" * 60)
print("TEST 2: scatter_update with DTensor")
print("=" * 60)

from torch.distributed._tensor import DTensor, DeviceMesh, Replicate, Shard
from keras.src.backend.torch.distribution_lib import is_dtensor

# Create a simple device mesh (simulating single process)
device_type = "cuda" if torch.cuda.is_available() else "cpu"
device_mesh = DeviceMesh(
    device_type=device_type,
    mesh=torch.tensor([0]),  # Single device
    mesh_dim_names=["model"]
)

# Create a DTensor input
local_tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
dtensor_input = DTensor.from_local(local_tensor, device_mesh, [Replicate()])

# Create regular indices and updates
indices = torch.tensor([[0, 1]], dtype=torch.int64)
updates = torch.tensor([[10.0, 20.0]], dtype=torch.float32)

print(f"DTensor input: {dtensor_input}")
print(f"Local tensor: {local_tensor}")
print(f"Indices: {indices}")
print(f"Updates: {updates}")

result = scatter_update(dtensor_input, indices, updates)

print(f"Result: {result}")
print(f"Result type: {type(result)}")
print(f"Is DTensor: {is_dtensor(result)}")

# Verify result is a DTensor with same shape
assert is_dtensor(result), "Result should be a DTensor"
assert result.shape == dtensor_input.shape, f"Shape mismatch: {result.shape} != {dtensor_input.shape}"

# Verify the update was applied correctly to local tensor
expected_local = torch.tensor([[10.0, 20.0, 3.0], [4.0, 5.0, 6.0]])
actual_local = result.to_local()
print(f"Expected local: {expected_local}")
print(f"Actual local: {actual_local}")
assert torch.allclose(actual_local, expected_local), f"Test 2 failed: {actual_local} != {expected_local}"
print("✓ Test 2 PASSED: scatter_update with DTensor works\n")


# Test 3: Verify the original test case scenario
print("=" * 60)
print("TEST 3: Simulate the error scenario from the original test")
print("=" * 60)

# This simulates the scenario where inputs could be a DTensor and updates a regular tensor
# The fix ensures this works by extracting local tensor, doing the update, and reconstructing DTensor

# Create a sharded DTensor (simulating model parallel scenario)
world_size = 2
if torch.cuda.is_available() or True:  # Use CPU for testing
    # Create mesh with 2 "devices" (ranks)
    mesh_tensor = torch.tensor([0, 1])
    device_mesh_2 = DeviceMesh(
        device_type="cpu",
        mesh=mesh_tensor,
        mesh_dim_names=["model"]
    )
    
    # Create a replicated DTensor
    local_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    dtensor = DTensor.from_local(local_tensor, device_mesh_2, [Replicate()])
    
    # Create indices and updates as regular tensors
    indices = torch.tensor([[0, 0]], dtype=torch.int64)  # Update row 0
    updates = torch.tensor([[9.0, 9.0]], dtype=torch.float32)  # With these values
    
    print(f"DTensor: {dtensor}")
    print(f"Local before: {dtensor.to_local()}")
    print(f"Indices: {indices}")
    print(f"Updates: {updates}")
    
    result = scatter_update(dtensor, indices, updates)
    
    print(f"Result: {result}")
    print(f"Local after: {result.to_local()}")
    
    expected_local = torch.tensor([[9.0, 9.0], [3.0, 4.0]])
    assert torch.allclose(result.to_local(), expected_local), "Test 3 failed"
    print("✓ Test 3 PASSED: Simulated error scenario works\n")


print("=" * 60)
print("ALL TESTS PASSED!")
print("=" * 60)

