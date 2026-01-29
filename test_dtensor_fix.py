#!/usr/bin/env python3
"""
Test script to verify DTensor fixes work correctly.

This tests:
1. DTensor tensor conversion utilities
2. Mixed torch.Tensor and DTensor operations
3. numpy.py function updates
4. core.py function updates

Usage:
    # Single process, multi-GPU:
    python test_dtensor_fix.py
    
    # Multi-process with torchrun:
    torchrun --nproc_per_node=2 python test_dtensor_fix.py
"""

import os
# Set backend before any keras imports
os.environ["KERAS_BACKEND"] = "torch"
os.environ["KERAS_DISTRIBUTION_DEBUG"] = "1"

import torch
import torch.distributed as dist
import numpy as np

# Initialize distributed if needed
local_rank = int(os.environ.get("LOCAL_RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))

if world_size > 1 and not dist.is_initialized():
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl" if torch.cuda.is_available() else "gloo",
        init_method="env://"
    )


def log(msg, rank=None):
    """Simple logging with rank identification."""
    current_rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
    if rank is not None and current_rank != rank:
        return
    prefix = f"[Rank {current_rank:02d}]" if world_size > 1 else ""
    print(f"{prefix} {msg}")


def test_dtensor_utilities():
    """Test the new DTensor utility functions."""
    log("")
    log("=" * 60)
    log("TEST 1: DTensor Utility Functions")
    log("=" * 60)
    
    from keras.src.backend.torch.distribution_lib import (
        DTENSOR_AVAILABLE,
        DTensor,
        Replicate,
        Shard,
        is_dtensor,
        ensure_dtensor,
        create_replicate_dtensor,
        get_dtensor_local,
        convert_tensors_to_dtensor,
    )
    
    if not DTENSOR_AVAILABLE:
        log("DTensor not available, skipping test")
        return False
    
    log(f"DTENSOR_AVAILABLE: {DTENSOR_AVAILABLE}")
    
    # Create a test tensor
    test_tensor = torch.randn(4, 8)
    log(f"Test tensor shape: {test_tensor.shape}")
    
    # Test is_dtensor
    assert not is_dtensor(test_tensor), "is_dtensor should return False for regular tensor"
    log("✓ is_dtensor correctly identifies regular tensor")
    
    # Test create_replicate_dtensor
    dtensor = create_replicate_dtensor(test_tensor)
    assert is_dtensor(dtensor), "create_replicate_dtensor should return a DTensor"
    log(f"✓ create_replicate_dtensor: global_shape={dtensor.shape}, local_shape={dtensor.to_local().shape}")
    
    # Test get_dtensor_local
    local = get_dtensor_local(dtensor)
    assert local.shape == test_tensor.shape, "get_dtensor_local should return tensor with same shape"
    log("✓ get_dtensor_local works correctly")
    
    # Test convert_tensors_to_dtensor
    t1, t2 = torch.randn(4, 8), torch.randn(4, 8)
    d1, d2 = convert_tensors_to_dtensor(t1, t2)
    assert is_dtensor(d1) and is_dtensor(d2), "convert_tensors_to_dtensor should convert both tensors"
    log("✓ convert_tensors_to_dtensor works correctly")
    
    return True


def test_numpy_dtensor_handling():
    """Test that numpy.py functions handle DTensors correctly."""
    log("")
    log("=" * 60)
    log("TEST 2: NumPy Function DTensor Handling")
    log("=" * 60)
    
    from keras.src.backend.torch.distribution_lib import (
        DTENSOR_AVAILABLE,
        DTensor,
        Replicate,
        create_replicate_dtensor,
    )
    from keras.src.backend.torch.numpy import add, subtract, multiply, matmul
    
    if not DTENSOR_AVAILABLE:
        log("DTensor not available, skipping test")
        return False
    
    # Create test tensors
    t1 = torch.randn(4, 8)
    t2 = torch.randn(4, 8)
    
    # Convert one to DTensor
    dt1 = create_replicate_dtensor(t1)
    
    # Test add with mixed tensors
    result = add(dt1, t2)
    assert result is not None, "add should return a result"
    log(f"✓ add(dtensor, tensor): result type={type(result).__name__}")
    
    # Test add with mixed tensors (reversed)
    result = add(t2, dt1)
    assert result is not None, "add should return a result"
    log(f"✓ add(tensor, dtensor): result type={type(result).__name__}")
    
    # Test subtract
    result = subtract(dt1, t2)
    assert result is not None, "subtract should return a result"
    log(f"✓ subtract(dtensor, tensor): result type={type(result).__name__}")
    
    # Test multiply
    result = multiply(dt1, t2)
    assert result is not None, "multiply should return a result"
    log(f"✓ multiply(dtensor, tensor): result type={type(result).__name__}")
    
    # Test matmul
    t3 = torch.randn(8, 16)
    dt3 = create_replicate_dtensor(t3)
    result = matmul(dt1, dt3)
    assert result is not None, "matmul should return a result"
    log(f"✓ matmul(dtensor, dtensor): result type={type(result).__name__}")
    
    return True


def test_core_dtensor_handling():
    """Test that core.py functions handle DTensors correctly."""
    log("")
    log("=" * 60)
    log("TEST 3: Core Function DTensor Handling")
    log("=" * 60)
    
    from keras.src.backend.torch.distribution_lib import (
        DTENSOR_AVAILABLE,
        DTensor,
        Replicate,
        create_replicate_dtensor,
    )
    from keras.src.backend.torch.core import (
        is_tensor,
        shape,
        cast,
        convert_to_numpy,
    )
    
    if not DTENSOR_AVAILABLE:
        log("DTensor not available, skipping test")
        return False
    
    # Create test DTensor
    test_tensor = torch.randn(4, 8)
    dtensor = create_replicate_dtensor(test_tensor)
    
    # Test is_tensor with DTensor
    assert is_tensor(dtensor), "is_tensor should return True for DTensor"
    log("✓ is_tensor correctly identifies DTensor")
    
    # Test shape with DTensor
    dtensor_shape = shape(dtensor)
    assert dtensor_shape == (4, 8), f"shape should return global shape, got {dtensor_shape}"
    log(f"✓ shape returns global shape: {dtensor_shape}")
    
    # Test cast with DTensor
    casted = cast(dtensor, "float32")
    assert is_tensor(casted), "cast should return a tensor"
    log(f"✓ cast works with DTensor: dtype={casted.dtype}")
    
    # Test convert_to_numpy with DTensor
    np_array = convert_to_numpy(dtensor)
    assert isinstance(np_array, np.ndarray), "convert_to_numpy should return numpy array"
    assert np_array.shape == (4, 8), f"convert_to_numpy should preserve shape, got {np_array.shape}"
    log(f"✓ convert_to_numpy works with DTensor: shape={np_array.shape}")
    
    return True


def test_model_creation_with_dtensor():
    """Test model creation with DTensor weights."""
    log("")
    log("=" * 60)
    log("TEST 4: Model Creation with DTensor")
    log("=" * 60)
    
    import keras
    from keras import layers
    from keras.distribution import DeviceMesh, LayoutMap, ModelParallel, list_devices
    
    # Skip if less than 2 GPUs
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if gpu_count < 2:
        log("Skipping test (need >= 2 GPUs)")
        return True
    
    devices = [f"cuda:{i}" for i in range(min(2, gpu_count))]
    log(f"Using devices: {devices}")
    
    # Create mesh and layout map
    mesh = DeviceMesh(
        shape=(1, len(devices)),
        axis_names=["batch", "model"],
        devices=devices
    )
    
    layout_map = LayoutMap(mesh)
    layout_map["dense.*kernel"] = (None, "model")
    layout_map["dense.*bias"] = ("model",)
    
    # Create distribution
    mp = ModelParallel(
        layout_map=layout_map,
        batch_dim_name="batch"
    )
    
    # Create model in scope
    with mp.scope():
        model = keras.Sequential([
            layers.Dense(256, activation="relu", input_shape=(64,)),
            layers.Dense(128, activation="relu"),
            layers.Dense(10)
        ])
    
    log(f"Model created with {model.count_params():,} parameters")
    
    # Check that weights are DTensors
    from keras.src.backend.torch.distribution_lib import is_dtensor
    
    all_dtensor = True
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'kernel'):
            kernel = layer.kernel
            kernel_val = kernel.value if hasattr(kernel, 'value') else kernel
            if is_dtensor(kernel_val):
                local_shape = kernel_val.to_local().shape
                global_shape = kernel_val.shape
                log(f"  {layer.name}.kernel: global={global_shape}, local={local_shape} [DTensor]")
                
                # Verify sharding
                if local_shape[1] < global_shape[1]:
                    log(f"    ✓ Sharded correctly")
                else:
                    log(f"    ✗ NOT sharded (should have smaller local dim)")
                    all_dtensor = False
            else:
                log(f"  {layer.name}.kernel: shape={kernel_val.shape} [Parameter - NOT DTensor]")
                all_dtensor = False
    
    if all_dtensor:
        log("✓ All weights are properly sharded DTensors")
    else:
        log("✗ Some weights are NOT DTensors")
    
    return all_dtensor


def main():
    """Run all tests."""
    log("")
    log("=" * 70)
    log("DTENSOR FIX VERIFICATION TEST")
    log("=" * 70)
    log("")
    log(f"PyTorch version: {torch.__version__}")
    log(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"GPU count: {torch.cuda.device_count()}")
    log(f"World size: {world_size}")
    
    results = []
    
    # Run tests
    results.append(("DTensor Utilities", test_dtensor_utilities()))
    results.append(("NumPy DTensor Handling", test_numpy_dtensor_handling()))
    results.append(("Core DTensor Handling", test_core_dtensor_handling()))
    results.append(("Model with DTensor", test_model_creation_with_dtensor()))
    
    # Summary
    log("")
    log("=" * 70)
    log("TEST SUMMARY")
    log("=" * 70)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        log(f"  {name}: {status}")
    
    log(f"\nTotal: {passed}/{total} tests passed")
    
    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

