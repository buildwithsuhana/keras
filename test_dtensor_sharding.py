#!/usr/bin/env python3
"""
Test script to verify DTensor sharding is working correctly.
This tests the core functionality of the distribution system.
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

# Now import keras
import keras
from keras import layers
from keras.distribution import (
    DeviceMesh,
    LayoutMap,
    ModelParallel,
    DataParallel,
    initialize,
    list_devices,
    distribution,
)
from keras.src.distribution import distribution_lib

def main():
    print("=" * 60)
    print("DTENSOR SHARDING TEST")
    print("=" * 60)
    
    # Initialize distribution
    initialize()
    
    # Check DTensor availability
    from keras.src.backend.torch.distribution_lib import DTENSOR_AVAILABLE
    print(f"\nDTENSOR_AVAILABLE: {DTENSOR_AVAILABLE}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Distributed initialized: {dist.is_initialized() if dist.is_available() else False}")
    
    if dist.is_initialized():
        print(f"  Rank: {dist.get_rank()}, World size: {dist.get_world_size()}")
    
    # Get devices
    devices = list_devices("gpu")
    print(f"\nAvailable devices: {devices}")
    
    if len(devices) < 2:
        print("Need at least 2 devices for this test. Skipping...")
        return
    
    # Create device mesh for model parallelism
    mesh = DeviceMesh(
        shape=(1, len(devices)),
        axis_names=["batch", "model"],
        devices=devices
    )
    print(f"\nDeviceMesh created: shape={mesh.shape}, axes={mesh.axis_names}")
    
    # Create layout map
    layout_map = LayoutMap(mesh)
    layout_map["dense.*kernel"] = (None, "model")  # Shard on model axis
    layout_map["dense.*bias"] = ("model",)
    print("LayoutMap configured:")
    for key in layout_map.keys():
        layout = layout_map[key]
        print(f"  - {key}: axes={layout.axes}")
    
    # Create ModelParallel distribution
    mp = ModelParallel(
        layout_map=layout_map,
        batch_dim_name="batch"
    )
    print(f"\nModelParallel created: {mp}")
    
    # Check mesh info before entering scope
    mesh_info_before = distribution_lib._get_mesh_info()
    print(f"Mesh info before scope: {mesh_info_before is not None}")
    
    # Create model within distribution scope
    with mp.scope():
        print("\n--- Inside distribution scope ---")
        
        # Check mesh info after entering scope
        mesh_info = distribution_lib._get_mesh_info()
        print(f"Mesh info after entering scope: {mesh_info is not None}")
        if mesh_info:
            print(f"  Shape: {mesh_info.get('shape')}")
            print(f"  Dim names: {mesh_info.get('dim_names')}")
        
        # Create a simple model
        model = keras.Sequential([
            layers.Dense(64, activation="relu", input_shape=(32,)),
            layers.Dense(10)
        ])
        
        print(f"\nModel created with {model.count_params()} parameters")
        
        # Check variable shapes
        for i, layer in enumerate(model.layers):
            if hasattr(layer, 'kernel'):
                kernel = layer.kernel
                kernel_val = kernel.value if hasattr(kernel, 'value') else kernel
                print(f"\nLayer {i} ({layer.name}):")
                print(f"  Kernel shape: {kernel_val.shape}")
                print(f"  Kernel type: {type(kernel_val).__name__}")
                
                # Check if it's a DTensor
                if hasattr(kernel_val, 'to_local'):
                    print(f"  Is DTensor: Yes")
                    local_shape = kernel_val.to_local().shape
                    print(f"  Local shape: {local_shape}")
                else:
                    print(f"  Is DTensor: No")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETED")
    print("=" * 60)
    
    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()

