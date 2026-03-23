import os
import sys
import torch.multiprocessing as mp

# DO NOT import keras or torch at top level to avoid premature CUDA initialization

def run_test(rank, world_size):
    # Set environment variables for this process - MUST BE FIRST
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    os.environ["KERAS_BACKEND"] = "torch"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29508"
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    
    # Now safe to import
    import torch
    import keras
    from keras import layers
    from keras.distribution import DeviceMesh, LayoutMap, ModelParallel, TensorLayout

    # Set device
    torch.cuda.set_device(0)
    
    # Initialize Keras distribution system
    print(f"Rank {rank} initializing distribution...")
    keras.distribution.initialize()
    print(f"Rank {rank} distribution initialized.")
    
    # Get available devices
    devices = keras.distribution.list_devices("gpu")
    print(f"Rank {rank}/{world_size} starting with local devices: {devices}")
    
    # Create a 1D device mesh for ModelParallel
    global_devices = [f"gpu:{i}" for i in range(world_size)]
    mesh = DeviceMesh(shape=(world_size,), axis_names=("model",), devices=global_devices)
    
    # Define layout map
    layout_map = LayoutMap(mesh)
    # Shard kernels along the "model" axis
    layout_map[".*dense_1/kernel"] = TensorLayout(axes=(None, "model"), device_mesh=mesh)
    layout_map[".*dense_2/kernel"] = TensorLayout(axes=("model", None), device_mesh=mesh)
    
    # Create the ModelParallel distribution strategy
    distribution = ModelParallel(layout_map=layout_map, batch_dim_name="model", auto_shard_dataset=False)
    
    with distribution.scope():
        # Simple MLP model
        print(f"Rank {rank} creating model...")
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
        import numpy as np
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
    # Determine world size
    import torch
    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()
    else:
        world_size = 2
        
    # Use 'spawn' to ensure a clean slate
    mp.set_start_method('spawn', force=True)
    try:
        mp.spawn(run_test, args=(world_size,), nprocs=world_size, join=True)
    except Exception as e:
        print(f"Test failed with error: {e}")
