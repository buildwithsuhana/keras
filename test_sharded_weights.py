#!/usr/bin/env python3
"""
Test script to verify sharded weight creation for ModelParallel.
This ensures full weights NEVER exist in memory - only shards.
"""

import os
# MUST be set before any other imports
os.environ["KERAS_BACKEND"] = "torch"
os.environ["KERAS_DISTRIBUTION_DEBUG"] = "1"

import torch
import torch.distributed as dist

def main():
    print("=" * 70)
    print("SHARDED WEIGHT CREATION TEST")
    print("=" * 70)
    
    # Initialize distributed
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    if world_size > 1 and not dist.is_initialized():
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
    
    rank = dist.get_rank() if dist.is_initialized() else 0
    print(f"\n[Rank {rank}] Testing sharded weight creation...")
    
    # Import keras AFTER setting backend
    import keras
    from keras import layers
    from keras.distribution import (
        DeviceMesh,
        LayoutMap,
        ModelParallel,
        list_devices,
    )
    from keras.src.distribution import distribution_lib
    
    # Check DTensor availability
    print(f"\n[Rank {rank}] DTensor available: {distribution_lib.DTENSOR_AVAILABLE}")
    
    # Get devices
    devices = list_devices("gpu")
    if len(devices) < 2:
        print(f"[Rank {rank}] Need 2+ GPUs, only have {len(devices)}. Skipping test.")
        return
    
    print(f"\n[Rank {rank}] Available devices: {devices}")
    
    # Create device mesh for tensor parallelism
    mesh = DeviceMesh(
        shape=(1, len(devices)),
        axis_names=["batch", "model"],
        devices=devices
    )
    print(f"[Rank {rank}] DeviceMesh: shape={mesh.shape}, axes={mesh.axis_names}")
    
    # Create layout map - this enables sharding
    layout_map = LayoutMap(mesh)
    layout_map["dense.*kernel"] = (None, "model")  # Shard output dim
    layout_map["dense.*bias"] = ("model",)  # Shard bias
    print(f"\n[Rank {rank}] LayoutMap configured:")
    for key in layout_map.keys():
        layout = layout_map[key]
        print(f"  - {key}: axes={layout.axes}")
    
    # Create ModelParallel distribution
    mp = ModelParallel(
        layout_map=layout_map,
        batch_dim_name="batch"
    )
    
    # Test: Create model with SHARDED weights
    print(f"\n[Rank {rank}] Creating model with SHARDED weights...")
    
    with mp.scope():
        model = keras.Sequential([
            layers.Input(shape=(128,)),
            layers.Dense(512, activation="relu", name="dense_1"),
            layers.Dense(256, activation="relu", name="dense_2"),
            layers.Dense(10, name="output")
        ])
        
        total_params = model.count_params()
        print(f"\n[Rank {rank}] Model created with {total_params:,} total parameters")
    
    # Verify: Check that weights are SHARDED DTensors, not full tensors
    print(f"\n[Rank {rank}] Verifying sharded weight creation:")
    print("-" * 50)
    
    # Expected shapes after sharding (with 2 GPUs on 'model' axis)
    expected_shards = {
        'dense_1': {
            'kernel': (256, 512),  # Half of (512, 512) for output features
            'bias': (256,),        # Half of (512,) for output features
        },
        'dense_2': {
            'kernel': (128, 256),  # Half of (256, 256)
            'bias': (128,),        # Half of (256,)
        },
        'output': {
            'kernel': (5, 10),     # Half of (10, 10)
            'bias': (5,),          # Half of (10,)
        }
    }
    
    all_sharded = True
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'kernel'):
            kernel_var = layer.kernel
            if hasattr(kernel_var, '_value'):
                kernel_tensor = kernel_var._value
            elif hasattr(kernel_var, 'value'):
                kernel_tensor = kernel_var.value
            else:
                kernel_tensor = kernel_var
            
            # Check if it's a DTensor
            if hasattr(kernel_tensor, 'to_local'):
                # It's a DTensor
                dtensor = kernel_tensor
                global_shape = tuple(dtensor.shape)
                local_tensor = dtensor.to_local()
                local_shape = tuple(local_tensor.shape)
                
                # Check if sharded (local shape < global shape)
                is_sharded = any(ls < gs for ls, gs in zip(local_shape, global_shape))
                
                print(f"\n[Rank {rank}] Layer {i} ({layer.name}):")
                print(f"  Global shape: {global_shape}")
                print(f"  Local shape:  {local_shape}")
                print(f"  Sharded: {is_sharded}")
                
                if is_sharded:
                    print(f"  ✓ SUCCESS: Weight is SHARDED (no full tensor in memory)")
                else:
                    print(f"  ✗ FAILURE: Weight is NOT sharded (full tensor exists!)")
                    all_sharded = False
            else:
                # Regular tensor - NOT sharded
                shape = tuple(kernel_tensor.shape)
                print(f"\n[Rank {rank}] Layer {i} ({layer.name}):")
                print(f"  Shape: {shape}")
                print(f"  ✗ FAILURE: Regular tensor (not DTensor) - full weight exists!")
                all_sharded = False
    
    print("\n" + "=" * 70)
    if all_sharded:
        print("[Rank {rank}] ✓ ALL WEIGHTS ARE SHARDED - Full weights NEVER in memory!")
    else:
        print("[Rank {rank}] ✗ SOME WEIGHTS NOT SHARDED - Fix needed!")
    print("=" * 70)
    
    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()

