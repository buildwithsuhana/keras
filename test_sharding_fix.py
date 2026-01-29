#!/usr/bin/env python3
"""
Test script to verify that DTensor sharding is working correctly after the fix.

This script tests:
1. Device mesh registration in global state
2. Manual sharding fallback when DTensor is not available
3. Variable sharding verification

Usage:
    # Single process, multi-GPU:
    python test_sharding_fix.py
    
    # Multi-process with torchrun:
    torchrun --nproc_per_node=2 python test_sharding_fix.py
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


def test_device_mesh_registration():
    """Test that device mesh is properly registered in global state."""
    log("=" * 60)
    log("TEST 1: Device Mesh Registration")
    log("=" * 60)
    
    from keras.distribution import DeviceMesh, list_devices
    from keras.src.backend.torch.distribution_lib import (
        _get_default_device_mesh,
        _get_mesh_info,
        DTENSOR_AVAILABLE
    )
    
    devices = list_devices("gpu")
    if not devices:
        devices = [f"cpu:{i}" for i in range(min(2, os.cpu_count() or 1))]
    
    log(f"Available devices: {devices}")
    
    # Create device mesh
    mesh = DeviceMesh(
        shape=(1, len(devices)),
        axis_names=["batch", "model"],
        devices=devices
    )
    
    log(f"DeviceMesh created: shape={mesh.shape}, axes={mesh.axis_names}")
    
    # Access backend_mesh to trigger creation and registration
    backend_mesh = mesh.backend_mesh
    log(f"Backend mesh type: {type(backend_mesh).__name__}")
    
    # Check if it's registered in global state
    registered_mesh = _get_default_device_mesh()
    mesh_info = _get_mesh_info()
    
    if registered_mesh is not None:
        log(f"✓ DeviceMesh registered in global state")
        log(f"  Mesh info: {mesh_info}")
        return True
    else:
        log("✗ DeviceMesh NOT found in global state")
        return False


def test_manual_sharding():
    """Test manual sharding fallback."""
    log("")
    log("=" * 60)
    log("TEST 2: Manual Sharding Fallback")
    log("=" * 60)
    
    from keras.distribution import DeviceMesh, TensorLayout, list_devices
    from keras.src.backend.torch.distribution_lib import (
        distribute_variable,
        _get_default_device_mesh,
    )
    from torch.distributed._tensor import DTensor
    
    devices = list_devices("gpu")
    if not devices:
        devices = ["cpu:0", "cpu:1"]
    
    log(f"Using devices: {devices}")
    
    # Create mesh and access backend_mesh to register it
    mesh = DeviceMesh(
        shape=(1, len(devices)),
        axis_names=["batch", "model"],
        devices=devices
    )
    _ = mesh.backend_mesh  # This triggers registration
    
    # Create layout for sharding on model axis
    layout = TensorLayout(axes=(None, "model"), device_mesh=mesh)
    
    log(f"Created layout: axes={layout.axes}")
    
    # Create a test tensor (simulating a kernel weight)
    # Shape: (128, 256) - will be sharded on dim 1 (256 -> 128 per device for 2 devices)
    test_tensor = np.random.random((128, 256)).astype("float32")
    
    log(f"Original tensor shape: {test_tensor.shape}")
    
    # Call distribute_variable
    param = distribute_variable(test_tensor, layout)
    
    log(f"Parameter type: {type(param).__name__}")
    log(f"Parameter shape: {param.shape}")
    
    # Check if it's a DTensor
    is_dtensor = isinstance(param, DTensor)
    log(f"Is DTensor: {is_dtensor}")
    
    if is_dtensor:
        local_shape = param.to_local().shape
        log(f"Local shape (on this device): {local_shape}")
        
        # Verify sharding
        expected_local_dim1 = 256 // len(devices)
        if local_shape[1] == expected_local_dim1:
            log(f"✓ DTensor sharding verified: dim 1 sharded from 256 to {local_shape[1]}")
            return True
        else:
            log(f"✗ DTensor sharding mismatch: expected {expected_local_dim1}, got {local_shape[1]}")
            return False
    else:
        # Check if manual sharding was applied
        expected_local_dim1 = 256 // len(devices)
        if param.shape[1] == expected_local_dim1:
            log(f"✓ Manual sharding verified: dim 1 sharded from 256 to {param.shape[1]}")
            return True
        else:
            log(f"✗ Manual sharding NOT applied: expected local dim {expected_local_dim1}, got {param.shape[1]}")
            return False


def test_model_creation_with_sharding():
    """Test model creation with sharding."""
    log("")
    log("=" * 60)
    log("TEST 3: Model Creation with Sharding")
    log("=" * 60)
    
    import keras
    from keras import layers
    from keras.distribution import DeviceMesh, LayoutMap, ModelParallel, list_devices
    
    devices = list_devices("gpu")
    if not devices:
        devices = [f"cpu:{i}" for i in range(min(2, os.cpu_count() or 1))]
    
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
    
    log("LayoutMap configured:")
    for key in layout_map.keys():
        layout = layout_map[key]
        log(f"  - {key}: axes={layout.axes}")
    
    # Create distribution
    mp = ModelParallel(
        layout_map=layout_map,
        batch_dim_name="batch"
    )
    
    log(f"ModelParallel created: {mp}")
    
    # Check mesh is registered
    from keras.src.backend.torch.distribution_lib import _get_default_device_mesh
    registered_mesh = _get_default_device_mesh()
    if registered_mesh is not None:
        log(f"✓ DeviceMesh registered before model creation")
    else:
        log("✗ DeviceMesh NOT registered before model creation")
    
    # Create model in scope
    with mp.scope():
        model = keras.Sequential([
            layers.Dense(256, activation="relu", input_shape=(64,)),
            layers.Dense(128, activation="relu"),
            layers.Dense(10)
        ])
        
        total_params = model.count_params()
        log(f"Model created with {total_params:,} parameters")
        
        # Check variable shapes
        log("")
        log("Variable shapes:")
        
        from torch.distributed._tensor import DTensor
        
        all_sharded = True
        for i, layer in enumerate(model.layers):
            if hasattr(layer, 'kernel'):
                kernel = layer.kernel
                kernel_val = kernel.value if hasattr(kernel, 'value') else kernel
                
                is_dtensor = isinstance(kernel_val, DTensor)
                
                if is_dtensor:
                    local_shape = kernel_val.to_local().shape
                    global_shape = kernel_val.shape
                    log(f"  {layer.name}.kernel: global={global_shape}, local={local_shape} [DTensor]")
                    
                    # Check if sharded
                    if local_shape[1] < global_shape[1]:
                        log(f"    ✓ Sharded on dim 1: {global_shape[1]} -> {local_shape[1]}")
                    else:
                        log(f"    ✗ NOT sharded")
                        all_sharded = False
                else:
                    log(f"  {layer.name}.kernel: shape={kernel_val.shape} [Parameter]")
                    # For 2 devices with model axis, should have half the columns
                    if kernel_val.shape[1] == 256 // len(devices):
                        log(f"    ✓ Manually sharded: {kernel_val.shape}")
                    else:
                        log(f"    ✗ NOT sharded (shape={kernel_val.shape}, expected {256 // len(devices)})")
                        all_sharded = False
    
    log("")
    if all_sharded:
        log("✓ All variables properly sharded")
        return True
    else:
        log("✗ Some variables NOT properly sharded")
        return False


def test_data_parallel_replication():
    """Test that DataParallel replicates variables (not shards)."""
    log("")
    log("=" * 60)
    log("TEST 4: DataParallel Variable Replication")
    log("=" * 60)
    
    import keras
    from keras import layers
    from keras.distribution import DataParallel, list_devices
    
    devices = list_devices("gpu")
    if not devices:
        devices = [f"cpu:{i}" for i in range(min(2, os.cpu_count() or 1))]
    
    log(f"Using devices: {devices}")
    
    # Create distribution
    dp = DataParallel()
    
    log(f"DataParallel created: {dp}")
    
    # Check mesh is registered
    from keras.src.backend.torch.distribution_lib import _get_default_device_mesh
    registered_mesh = _get_default_device_mesh()
    if registered_mesh is not None:
        log(f"✓ DeviceMesh registered before model creation")
    else:
        log("✗ DeviceMesh NOT registered before model creation")
    
    # Create model in scope
    with dp.scope():
        model = keras.Sequential([
            layers.Dense(256, activation="relu", input_shape=(64,)),
            layers.Dense(128, activation="relu"),
            layers.Dense(10)
        ])
        
        total_params = model.count_params()
        log(f"Model created with {total_params:,} parameters")
        
        # Check variable shapes - should be replicated (same as global)
        log("")
        log("Variable shapes (should be replicated):")
        
        from torch.distributed._tensor import DTensor
        
        all_replicated = True
        for layer in model.layers:
            if hasattr(layer, 'kernel'):
                kernel = layer.kernel
                kernel_val = kernel.value if hasattr(kernel, 'value') else kernel
                
                is_dtensor = isinstance(kernel_val, DTensor)
                
                if is_dtensor:
                    local_shape = kernel_val.to_local().shape
                    global_shape = kernel_val.shape
                    log(f"  {layer.name}.kernel: global={global_shape}, local={local_shape} [DTensor]")
                    
                    # For DataParallel, local should equal global (replicated)
                    if local_shape == global_shape:
                        log(f"    ✓ Replicated (local == global)")
                    else:
                        log(f"    ✗ NOT replicated (sharded)")
                        all_replicated = False
                else:
                    log(f"  {layer.name}.kernel: shape={kernel_val.shape} [Parameter]")
                    # Should have full shape (replicated)
                    expected_shape = (64, 256) if "dense" in layer.name and "kernel" in layer.name else None
                    if expected_shape and kernel_val.shape == expected_shape:
                        log(f"    ✓ Replicated")
    
    log("")
    if all_replicated:
        log("✓ All variables properly replicated")
        return True
    else:
        log("✗ Some variables NOT properly replicated")
        return False


def main():
    """Run all tests."""
    log("")
    log("=" * 70)
    log("DTENSOR SHARDING FIX VERIFICATION TEST")
    log("=" * 70)
    log("")
    log(f"PyTorch version: {torch.__version__}")
    log(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"GPU count: {torch.cuda.device_count()}")
    log(f"World size: {world_size}")
    
    results = []
    
    # Run tests
    results.append(("Device Mesh Registration", test_device_mesh_registration()))
    results.append(("Manual Sharding", test_manual_sharding()))
    results.append(("ModelParallel Sharding", test_model_creation_with_sharding()))
    results.append(("DataParallel Replication", test_data_parallel_replication()))
    
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

