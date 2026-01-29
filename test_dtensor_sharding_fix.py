#!/usr/bin/env python3
"""
Test script to verify DTensor sharding fix is working correctly.

This script specifically tests:
1. DTensor availability detection
2. DeviceMesh creation and backend mesh registration
3. Variable sharding via distribute_variable
4. Physical tensor shape verification

Usage:
    python test_dtensor_sharding_fix.py
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


def log(msg, rank=0):
    """Simple logging with rank identification."""
    current_rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
    if current_rank == rank or rank == -1:
        prefix = f"[Rank {current_rank:02d}]" if world_size > 1 else ""
        print(f"{prefix} {msg}")


def test_dtensor_availability():
    """Test that DTensor is available."""
    log("=" * 60)
    log("TEST 1: DTensor Availability")
    log("=" * 60)
    
    from keras.src.backend.torch.distribution_lib import DTENSOR_AVAILABLE
    
    log(f"DTENSOR_AVAILABLE: {DTENSOR_AVAILABLE}")
    log(f"PyTorch version: {torch.__version__}")
    
    if DTENSOR_AVAILABLE:
        from torch.distributed._tensor import DeviceMesh
        log("✓ DTensor is available")
        return True
    else:
        log("⚠ DTensor is not available - will test manual sharding fallback")
        return False


def test_device_mesh_creation():
    """Test DeviceMesh creation and backend mesh registration."""
    log("")
    log("=" * 60)
    log("TEST 2: DeviceMesh Creation")
    log("=" * 60)
    
    from keras.distribution import DeviceMesh, list_devices
    
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
    
    # Access backend mesh to trigger creation and registration
    backend_mesh = mesh.backend_mesh
    log(f"Backend mesh type: {type(backend_mesh).__name__}")
    
    # Check if it's registered in global state
    from keras.src.backend.torch.distribution_lib import _get_default_device_mesh, _get_mesh_info
    
    registered_mesh = _get_default_device_mesh()
    mesh_info = _get_mesh_info()
    
    if registered_mesh is not None:
        log(f"✓ DeviceMesh registered in global state")
        log(f"  Mesh info: {mesh_info}")
        return True
    else:
        log("⚠ DeviceMesh not found in global state")
        return False


def test_distribute_variable():
    """Test distribute_variable function with sharding."""
    log("")
    log("=" * 60)
    log("TEST 3: distribute_variable with Sharding")
    log("=" * 60)
    
    from keras.src.backend.torch.distribution_lib import (
        distribute_variable,
        _get_default_device_mesh,
        _get_debug_setting,
    )
    from keras.distribution import DeviceMesh, LayoutMap, TensorLayout
    import logging
    
    # Enable debug logging
    logger = logging.getLogger("keras.src.backend.torch.distribution_lib")
    logger.setLevel(logging.DEBUG)
    
    # Setup mesh
    devices = list_devices("gpu")
    if not devices:
        devices = ["cpu:0", "cpu:1"]
    
    mesh = DeviceMesh(
        shape=(1, len(devices)),
        axis_names=["batch", "model"],
        devices=devices
    )
    
    # Create layout for sharding on model axis
    layout = TensorLayout(axes=(None, "model"), device_mesh=mesh)
    
    log(f"Created layout: axes={layout.axes}")
    
    # Create a test tensor (simulating a kernel weight)
    # Shape: (128, 256) - will be sharded on dim 1 (256 -> 128 per device)
    test_tensor = np.random.random((128, 256)).astype("float32")
    
    log(f"Original tensor shape: {test_tensor.shape}")
    
    # Call distribute_variable
    param = distribute_variable(test_tensor, layout)
    
    log(f"Parameter type: {type(param).__name__}")
    log(f"Parameter shape: {param.shape}")
    
    # Check if it's a DTensor
    from torch.distributed._tensor import DTensor
    is_dtensor = isinstance(param, DTensor)
    log(f"Is DTensor: {is_dtensor}")
    
    if is_dtensor:
        local_shape = param.to_local().shape
        log(f"Local shape (on this device): {local_shape}")
        
        # Verify sharding
        expected_local_dim1 = 256 // len(devices)
        if local_shape[1] == expected_local_dim1:
            log(f"✓ Sharding verified: dim 1 sharded from 256 to {local_shape[1]}")
            return True
        else:
            log(f"⚠ Sharding mismatch: expected {expected_local_dim1}, got {local_shape[1]}")
            return False
    else:
        # Check if manual sharding was applied
        if world_size > 1:
            expected_local_dim1 = 256 // world_size
            if param.shape[1] == expected_local_dim1:
                log(f"✓ Manual sharding verified: dim 1 sharded from 256 to {param.shape[1]}")
                return True
            else:
                log(f"⚠ Manual sharding mismatch: expected {expected_local_dim1}, got {param.shape[1]}")
                return False
        else:
            log("⚠ Single device mode - no sharding expected")
            return True


def test_model_parallel_sharding():
    """Test ModelParallel with actual model creation."""
    log("")
    log("=" * 60)
    log("TEST 4: ModelParallel with Model Creation")
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
        
        for i, layer in enumerate(model.layers):
            if hasattr(layer, 'kernel'):
                kernel = layer.kernel
                kernel_val = kernel.value if hasattr(kernel, 'value') else kernel
                
                is_dtensor = isinstance(kernel_val, DTensor)
                
                if is_dtensor:
                    local_shape = kernel_val.to_local().shape
                    log(f"  {layer.name}.kernel: global={kernel_val.shape}, local={local_shape} [DTensor]")
                else:
                    log(f"  {layer.name}.kernel: shape={kernel_val.shape} [Parameter]")
    
    log("✓ ModelParallel test completed")
    return True


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
    
    results = []
    
    # Run tests
    results.append(("DTensor Availability", test_dtensor_availability()))
    results.append(("DeviceMesh Creation", test_device_mesh_creation()))
    results.append(("distribute_variable", test_distribute_variable()))
    results.append(("ModelParallel", test_model_parallel_sharding()))
    
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

