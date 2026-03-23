import os
import torch

# SET BACKEND BEFORE IMPORTING KERAS
os.environ["KERAS_BACKEND"] = "torch"
# Detect if GPU is available
device_type = "gpu" if torch.cuda.is_available() else "cpu"
keras_device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["KERAS_TORCH_DEVICE"] = keras_device

import numpy as np
import keras
from keras import layers
from keras.distribution import DeviceMesh, LayoutMap, ModelParallel, TensorLayout

# For this test, we simulate multi-process distribution
def run_test(rank, world_size):
    # Configure environment for each process
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29508"
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["KERAS_TORCH_DEVICE"] = keras_device
    
    # Initialize Keras distribution system
    keras.distribution.initialize()
    
    # Get available devices
    devices = keras.distribution.list_devices(device_type)
    print(f"Rank {rank}/{world_size} starting with devices: {devices}")
    
    # Create a 1D device mesh for ModelParallel
    mesh = DeviceMesh(shape=(world_size,), axis_names=("model",), devices=devices[:world_size])
    
    # Define layout map
    layout_map = LayoutMap(mesh)
    # Shard kernels along the "model" axis
    layout_map[".*dense_1/kernel"] = TensorLayout(axes=(None, "model"), device_mesh=mesh)
    layout_map[".*dense_2/kernel"] = TensorLayout(axes=("model", None), device_mesh=mesh)
    
    # Create the ModelParallel distribution strategy
    distribution = ModelParallel(layout_map=layout_map, batch_dim_name="model", auto_shard_dataset=False)
    
    with distribution.scope():
        # Simple MLP model
        model = keras.Sequential([
            layers.Dense(128, activation="relu", name="dense_1"),
            layers.Dense(64, name="dense_2")
        ])
        
        # Build the model
        model.build((None, 64))
        
        # Verify sharding
        kernel_1 = model.get_layer("dense_1").kernel
        from torch.distributed.tensor import DTensor
        if isinstance(kernel_1.value, DTensor):
            local_shape = kernel_1.value.to_local().shape
            print(f"Rank {rank}, dense_1 kernel local shape: {local_shape}")
            # Global (64, 128), sharded on axis 1 (model) -> (64, 64)
            assert local_shape == (64, 64)
            
        kernel_2 = model.get_layer("dense_2").kernel
        if isinstance(kernel_2.value, DTensor):
            local_shape = kernel_2.value.to_local().shape
            print(f"Rank {rank}, dense_2 kernel local shape: {local_shape}")
            # Global (128, 64), sharded on axis 0 (model) -> (64, 64)
            assert local_shape == (64, 64)

        # Compile the model
        model.compile(optimizer="sgd", loss="mse")
        
        # Generate dummy data
        batch_size = 8
        x_in = np.random.randn(batch_size, 64).astype("float32")
        y_in = np.random.randn(batch_size, 64).astype("float32")
        
        # Execute fit
        print(f"Rank {rank} starting fit...")
        model.fit(x_in, y_in, epochs=1, batch_size=4)
        
        if rank == 0:
            print("Successfully finished model.fit with ModelParallel on MLP!")

    # Cleanup
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    # Simulate devices using processes
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 2
    import torch.multiprocessing as mp
    try:
        mp.spawn(run_test, args=(world_size,), nprocs=world_size, join=True)
    except Exception as e:
        print(f"Test failed with error: {e}")
