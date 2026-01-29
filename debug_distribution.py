#!/usr/bin/env python3
"""
Debug script to trace where distribution() returns None in distribute_variable.
"""

import os
# Set backend before any other imports
os.environ["KERAS_BACKEND"] = "torch"
os.environ["KERAS_DISTRIBUTION_DEBUG"] = "1"

import torch
import torch.distributed as dist

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


def main():
    log("")
    log("=" * 70)
    log("DEBUG: Trace distribution() in distribute_variable")
    log("=" * 70)
    log("")
    
    import keras
    from keras import layers
    from keras.distribution import DeviceMesh, LayoutMap, ModelParallel, list_devices
    import numpy as np
    from keras.src.distribution.distribution_lib import distribution, set_distribution
    
    # Check GPU count
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
    else:
        gpu_count = 1
    
    if gpu_count < 2:
        log("Need >= 2 GPUs for this test")
        return
    
    devices = list_devices("gpu")
    log(f"Devices: {devices}")
    
    # Create device mesh
    mesh = DeviceMesh(
        shape=(1, len(devices)),
        axis_names=["batch", "model"],
        devices=devices
    )
    log(f"DeviceMesh created: shape={mesh.shape}")
    
    # Create layout map
    layout_map = LayoutMap(mesh)
    layout_map["dense.*kernel"] = (None, "model")
    layout_map["dense.*bias"] = ("model",)
    log("LayoutMap configured")
    
    # Create distribution
    mp = ModelParallel(
        layout_map=layout_map,
        batch_dim_name="batch",
        auto_shard_dataset=False
    )
    log(f"ModelParallel created")
    
    # Test 1: Check distribution() BEFORE scope
    log("")
    log("=" * 70)
    log("TEST 1: Check distribution() BEFORE entering scope")
    log("=" * 70)
    dist_before = distribution()
    log(f"distribution() before scope: {dist_before}")
    
    # Test 2: Check distribution() AFTER entering scope
    log("")
    log("=" * 70)
    log("TEST 2: Check distribution() INSIDE scope")
    log("=" * 70)
    
    with mp.scope():
        dist_inside = distribution()
        log(f"distribution() inside scope: {dist_inside}")
        log(f"distribution is mp: {dist_inside is mp}")
        
        if dist_inside is None:
            log("ERROR: distribution() is None inside scope!")
            log("This is the bug - distribution not being set properly")
        else:
            log("SUCCESS: distribution is set correctly")
            
            # Now try creating a model
            log("")
            log("Creating model...")
            
            # Monkey-patch distribute_variable to trace it
            from keras.src.backend.torch import distribution_lib as torch_dist_lib
            
            original_distribute_variable = torch_dist_lib.distribute_variable
            
            def traced_distribute_variable(tensor, layout):
                from keras.src.distribution.distribution_lib import distribution as get_dist
                d = get_dist()
                log(f"  [distribute_variable] distribution() = {d}")
                log(f"  [distribute_variable] layout = {layout}")
                return original_distribute_variable(tensor, layout)
            
            torch_dist_lib.distribute_variable = traced_distribute_variable
            
            model = keras.Sequential([
                layers.Dense(256, activation="relu", input_shape=(128,)),
                layers.Dense(10)
            ])
            
            # Restore
            torch_dist_lib.distribute_variable = original_distribute_variable
            
            log("Model created successfully")
            
            # Check variable shapes
            log("")
            log("Variable shapes:")
            for layer in model.layers:
                if hasattr(layer, 'kernel'):
                    kernel = layer.kernel
                    kernel_val = kernel.value if hasattr(kernel, 'value') else kernel
                    log(f"  {layer.name}.kernel: shape={kernel_val.shape}")
                    # Check if it's a DTensor
                    if hasattr(kernel_val, 'to_local'):
                        local_shape = kernel_val.to_local().shape
                        log(f"    DTensor local shape: {local_shape}")
                        if local_shape[1] < kernel_val.shape[1]:
                            log(f"    ✓ SHARDED (global={kernel_val.shape[1]}, local={local_shape[1]})")
                        else:
                            log(f"    ✗ NOT sharded")
                    else:
                        log(f"    Regular Parameter (no DTensor)")
    
    # Test 3: Check distribution() AFTER scope
    log("")
    log("=" * 70)
    log("TEST 3: Check distribution() AFTER exiting scope")
    log("=" * 70)
    dist_after = distribution()
    log(f"distribution() after scope: {dist_after}")
    
    log("")
    log("=" * 70)
    log("SUMMARY")
    log("=" * 70)
    log(f"Before scope: distribution() = {dist_before}")
    log(f"Inside scope: distribution() = {dist_inside}")
    log(f"After scope:  distribution() = {dist_after}")
    
    if dist_before is None and dist_inside is not None and dist_after is None:
        log("✓ Distribution context manager working correctly")
    elif dist_before is None and dist_inside is None:
        log("✗ BUG: distribution() returns None even inside scope!")
    else:
        log(f"? Unexpected state")
    
    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

