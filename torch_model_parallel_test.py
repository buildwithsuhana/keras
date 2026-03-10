import os

os.environ["KERAS_BACKEND"] = "torch"

import torch
import numpy as np
import keras
from keras.distribution import DeviceMesh, LayoutMap, ModelParallel, set_distribution

def test_model_parallel():
    # Initialize distribution for single process
    keras.distribution.initialize()
    
    # 1. Setup DeviceMesh
    devices = ["cpu:0"] 
    mesh = DeviceMesh(shape=(1,), axis_names=("model",), devices=devices)
    layout_map = LayoutMap(mesh)
    layout_map[".*dense.*kernel"] = ("model", None)
    
    distribution = ModelParallel(layout_map=layout_map)
    
    with distribution.scope():
        model = keras.Sequential([
            keras.layers.Dense(8, input_shape=(4,), name="dense")
        ])
        model.build()
        
        # Check if the kernel is sharded
        kernel = model.layers[0].kernel
        print(f"Kernel path: {kernel.path}")
        print(f"Kernel value type: {type(kernel.value)}")
        
        # Try a forward pass
        x = torch.randn(2, 4)
        # Distribute the input tensor
        from keras.src.distribution.distribution_lib import distribute_tensor
        x = distribute_tensor(x, distribution.get_data_layout(x.shape))
        y = model(x)
        print(f"Output shape: {y.shape}")
        print(f"Output type: {type(y)}")

if __name__ == "__main__":
    try:
        test_model_parallel()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
