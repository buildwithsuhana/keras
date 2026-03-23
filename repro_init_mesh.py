
import os
os.environ["KERAS_BACKEND"] = "torch"
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12358"

import torch
from torch.distributed.device_mesh import init_device_mesh
import keras
from keras.distribution import DeviceMesh, ModelParallel, LayoutMap
from keras.src.backend.torch.core import convert_to_tensor
from keras.src.backend.torch.distribution_lib import sharding_scope

def test_redundant_init_device_mesh():
    mesh = DeviceMesh(shape=(1,), axis_names=("model",), devices=["cpu:0"])
    layout_map = LayoutMap(mesh)
    dist = ModelParallel(layout_map=layout_map)
    
    with dist.scope():
        with sharding_scope():
            print("Starting multiple convert_to_tensor calls...")
            for i in range(100):
                x = convert_to_tensor([1.0, 2.0])
                if i % 10 == 0:
                    print(f"Iteration {i}")
            print("Finished successfully!")

if __name__ == "__main__":
    test_redundant_init_device_mesh()
