
import os

# Set environment variables for torch.distributed
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12345"
os.environ["KERAS_BACKEND"] = "torch"
os.environ["KERAS_DISTRIBUTION_DEBUG"] = "1"

import torch
import numpy as np
import keras
print(f"DEBUG | Keras backend: {keras.backend.backend()}")
from keras.src.distribution.distribution_lib import DeviceMesh, ModelParallel, set_distribution, LayoutMap

def reproduce():
    # 1. Setup a fake DeviceMesh and ModelParallel distribution
    print("Initializing DeviceMesh...")
    mesh = DeviceMesh(shape=(1, 1), axis_names=["batch", "model"], devices=["cpu:0"])
    dist = ModelParallel(device_mesh=mesh, layout_map=LayoutMap(mesh))
    
    print("Setting distribution to ModelParallel...")
    set_distribution(dist)
    
    try:
        # 2. Create a functional model with Embedding
        print("Creating functional model with Embedding layer...")
        inputs = keras.Input(shape=(5,), dtype="int32")
        outputs = keras.layers.Embedding(100, 10)(inputs)
        model = keras.Model(inputs, outputs)
        print("✓ Model created successfully!")
        
        print("Attempting to build the model...")
        model.build((None, 5))
        print("✓ Model built successfully!")
        
        # 3. Try to call it on some data
        x = np.random.randint(0, 100, size=(2, 5)).astype("int32")
        y = model(x)
        print(f"✓ Model called successfully! Output shape: {y.shape}")
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        set_distribution(None)
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

if __name__ == "__main__":
    reproduce()
