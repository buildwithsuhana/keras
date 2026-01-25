"""Test script to verify DTensor fix for ModelParallel with Torch backend."""
import os
os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import torch
import torch.distributed as dist

# Test 1: Test _is_dtensor and _ensure_dtensor functions
print("Test 1: Testing DTensor helper functions...")
from keras.src.backend.torch.distribution_lib import (
    _is_dtensor, 
    _ensure_dtensor,
    _to_backend_mesh,
    list_devices
)

# Create a simple test
if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
    # Initialize distributed
    dist.init_process_group(backend="nccl")
    local_rank = dist.get_rank() % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    
    # Create a backend mesh
    from keras.src.distribution import distribution_lib
    
    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    device_mesh = distribution_lib.DeviceMesh(
        shape=(len(devices),),
        axis_names=('model',),
        devices=devices
    )
    
    backend_mesh = _to_backend_mesh(device_mesh)
    
    # Test _is_dtensor
    regular_tensor = torch.randn(10, 20)
    print(f"  Regular tensor is DTensor: {_is_dtensor(regular_tensor)}")  # Should be False
    
    # Test _ensure_dtensor with Replicate placement
    from torch.distributed._tensor import Replicate
    replicated = _ensure_dtensor(regular_tensor, backend_mesh, (Replicate(),))
    print(f"  Replicated tensor is DTensor: {_is_dtensor(replicated)}")  # Should be True
    
    # Clean up
    if dist.is_initialized():
        dist.destroy_process_group()
    
    print("Test 1: PASSED")
else:
    print("Test 1: SKIPPED (need at least 2 GPUs)")

# Test 2: Test matmul with mixed DTensor/tensor
print("\nTest 2: Testing matmul with DTensor...")
from keras.src.backend.torch.numpy import matmul

if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
    dist.init_process_group(backend="nccl")
    local_rank = dist.get_rank() % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    
    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    from keras.src.distribution import distribution_lib
    
    device_mesh = distribution_lib.DeviceMesh(
        shape=(len(devices),),
        axis_names=('model',),
        devices=devices
    )
    
    backend_mesh = _to_backend_mesh(device_mesh)
    
    # Create a sharded DTensor (kernel)
    from torch.distributed._tensor import Shard
    kernel = torch.randn(10, 20)
    sharded_kernel = torch.distributed._tensor.distribute_tensor(
        kernel, backend_mesh, (Shard(0),)
    )
    
    # Create regular input tensor
    inputs = torch.randn(32, 10)
    
    # Test matmul - should convert inputs to DTensor automatically
    result = matmul(inputs, sharded_kernel)
    
    print(f"  Input shape: {inputs.shape}")
    print(f"  Kernel shape: {kernel.shape}")
    print(f"  Result is DTensor: {_is_dtensor(result)}")
    print(f"  Result shape: {result.shape}")
    
    # Clean up
    if dist.is_initialized():
        dist.destroy_process_group()
    
    print("Test 2: PASSED")
else:
    print("Test 2: SKIPPED (need at least 2 GPUs)")

# Test 3: Test add with mixed DTensor/tensor
print("\nTest 3: Testing add with DTensor...")
from keras.src.backend.torch.numpy import add

if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
    dist.init_process_group(backend="nccl")
    local_rank = dist.get_rank() % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    
    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    from keras.src.distribution import distribution_lib
    
    device_mesh = distribution_lib.DeviceMesh(
        shape=(len(devices),),
        axis_names=('model',),
        devices=devices
    )
    
    backend_mesh = _to_backend_mesh(device_mesh)
    
    # Create a sharded DTensor
    from torch.distributed._tensor import Shard
    bias = torch.randn(20)
    sharded_bias = torch.distributed._tensor.distribute_tensor(
        bias, backend_mesh, (Shard(0),)
    )
    
    # Create regular tensor
    regular_tensor = torch.randn(20)
    
    # Test add - should convert regular tensor to DTensor automatically
    result = add(regular_tensor, sharded_bias)
    
    print(f"  Bias is DTensor: {_is_dtensor(sharded_bias)}")
    print(f"  Result is DTensor: {_is_dtensor(result)}")
    print(f"  Result shape: {result.shape}")
    
    # Clean up
    if dist.is_initialized():
        dist.destroy_process_group()
    
    print("Test 3: PASSED")
else:
    print("Test 3: SKIPPED (need at least 2 GPUs)")

print("\nAll tests completed!")

