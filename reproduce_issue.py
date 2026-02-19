
import os
import torch
import numpy as np
import keras
from keras.src.distribution.distribution_lib import DeviceMesh, ModelParallel, set_distribution, LayoutMap

# Mock environment for ModelParallel
os.environ["KERAS_BACKEND"] = "torch"

def reproduce():
    # 1. Setup a fake DeviceMesh and ModelParallel distribution
    # We use CPU devices for testing if no GPUs are available
    mesh = DeviceMesh(shape=(1, 1), axis_names=["batch", "model"], devices=["cpu:0"])
    dist = ModelParallel(device_mesh=mesh, layout_map=LayoutMap(mesh))
    
    print("Setting distribution to ModelParallel...")
    set_distribution(dist)
    
    try:
        # 2. Create a simple model/layer
        model = keras.Sequential([
            keras.layers.Dense(10, input_shape=(5,))
        ])
        
        # 3. Try to build it (this calls compute_output_spec -> symbolic_call)
        print("Attempting to build the model...")
        model.build()
        print("✓ Model built successfully!")
        
        # 4. Try to call it on some data
        x = np.random.random((2, 5)).astype("float32")
        y = model(x)
        print(f"✓ Model called successfully! Output shape: {y.shape}")
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        set_distribution(None)

if __name__ == "__main__":
    reproduce()
