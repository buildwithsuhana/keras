"""Quick test to verify the DTensor sharding fix is working"""

import os
os.environ["KERAS_BACKEND"] = "torch"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.distributed as dist
import keras
import keras_hub
from keras.src.distribution import ModelParallel, DeviceMesh, LayoutMap, list_devices, initialize

def run_quick_test():
    """Quick test to verify DTensor sharding."""
    initialize()
    
    devices = list_devices("gpu")
    num_devices = len(devices)
    
    if num_devices < 2:
        print("Need at least 2 GPUs")
        return
    
    mesh = DeviceMesh(
        shape=(1, 2),
        axis_names=["data", "model"],
        devices=devices
    )
    
    layout_map = LayoutMap(mesh)
    layout_map[".*kernel"] = (None, "model")
    
    strategy = ModelParallel(
        layout_map=layout_map,
        batch_dim_name="data",
        auto_shard_dataset=False
    )
    
    with strategy.scope():
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(64,)),
            keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    # Check if any weights are DTensors
    sharded_found = False
    for layer in model.layers:
        for weight in layer.weights:
            weight_value = weight.value if hasattr(weight, 'value') else weight
            if isinstance(weight_value, torch.distributed._tensor.DTensor):
                local_shape = tuple(weight_value.to_local().shape)
                global_shape = tuple(weight_value.shape)
                if local_shape != global_shape:
                    print(f"✓ SHARDED: {weight.name if hasattr(weight, 'name') else 'unnamed'}")
                    print(f"  Global: {global_shape}, Local: {local_shape}")
                    sharded_found = True
    
    if sharded_found:
        print("\n✓ SUCCESS: DTensor sharding is working!")
    else:
        print("\n✗ No sharded weights found")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    run_quick_test()

