import os
import torch

# Configure environment for distributed training and to avoid conflicts
os.environ["KERAS_BACKEND"] = "torch"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # Quiet TF logs
# Detect if GPU is available
device_type = "gpu" if torch.cuda.is_available() else "cpu"
keras_device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["KERAS_TORCH_DEVICE"] = keras_device

import numpy as np
import keras
import keras_hub
from keras.distribution import DeviceMesh, LayoutMap, ModelParallel, TensorLayout
from keras.src.backend.torch import distribution_lib as torch_dist_lib

# For this test, we simulate multi-process distribution
def run_test(rank, world_size):
    # Configure environment for each process
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29513"
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["KERAS_TORCH_DEVICE"] = keras_device
    # Force GPU to specific device
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
        torch.cuda.set_device(0) # In the process's view, it only has one GPU
    
    # Initialize Keras distribution system
    print(f"Rank {rank} initializing distribution...")
    keras.distribution.initialize()
    
    # Get available devices
    devices = keras.distribution.list_devices(device_type)
    print(f"Rank {rank}/{world_size} starting with devices: {devices}")
    
    # Create a 1D device mesh for ModelParallel
    mesh = DeviceMesh(shape=(world_size,), axis_names=("model",), devices=devices[:world_size])
    
    # Define layout map
    layout_map = LayoutMap(mesh)
    layout_map[".*dense.*kernel"] = TensorLayout(axes=(None, "model"), device_mesh=mesh)
    layout_map[".*embeddings/embeddings"] = TensorLayout(axes=("model", None), device_mesh=mesh)
    
    # Create the ModelParallel distribution strategy
    distribution = ModelParallel(layout_map=layout_map, batch_dim_name="model", auto_shard_dataset=False)
    
    with distribution.scope():
        # Load the OPT 125M model from Keras Hub
        print(f"Rank {rank} loading model...")
        model = keras_hub.models.OPTCausalLM.from_preset("opt_125m_en", load_weights=False)
        print(f"Rank {rank} model loaded.")
        
        # Compile the model
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            weighted_metrics=["accuracy"]
        )
        
        # Build the model manually by calling it with sharded data
        print(f"Rank {rank} building model...")
        x_str = ["Keras Hub is great.", "Distributed training works with ModelParallel!"] * 2
        preprocessor = model.preprocessor
        processed_data = preprocessor(x_str[:2]) # Get a small batch
        
        if isinstance(processed_data, tuple):
            inputs = processed_data[0]
        else:
            inputs = processed_data
            
        # Manually shard inputs
        def shard_input(tensor):
            if isinstance(tensor, torch.Tensor):
                layout = distribution.get_data_layout(tensor.shape)
                return torch_dist_lib.distribute_data_input(tensor, layout, distribution.batch_dim_name)
            return tensor
            
        from keras.src import tree
        sharded_inputs = tree.map_structure(shard_input, inputs)
        
        # Forward pass to build
        with torch_dist_lib.sharding_scope():
            model(sharded_inputs)
        print(f"Rank {rank} model built successfully.")
        
        # Execute fit
        print(f"Rank {rank} starting fit...")
        model.fit(x_str, x_str, epochs=1, batch_size=2)
        
        if rank == 0:
            print("Successfully finished model.fit on OPT-125M with ModelParallel!")

    # Cleanup
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    # Pre-download the preset to avoid multiple processes colliding
    print("Pre-downloading preset...")
    keras_hub.models.OPTCausalLM.from_preset("opt_125m_en", load_weights=False)
    
    # Simulate devices using processes
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 2
    import torch.multiprocessing as mp
    try:
        mp.spawn(run_test, args=(world_size,), nprocs=world_size, join=True)
    except Exception as e:
        print(f"Test failed with error: {e}")
