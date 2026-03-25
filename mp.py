import os
import sys
import torch.multiprocessing as mp

# DO NOT import keras or torch at top level to avoid premature CUDA initialization

def run_test(rank, world_size):
    # Set environment variables for this process - MUST BE FIRST
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    os.environ["KERAS_BACKEND"] = "torch"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29513"
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    
    # Now safe to import
    import torch
    import keras
    import keras_hub
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
    # For DTensor, we need the global view of devices
    global_devices = [f"gpu:{i}" for i in range(world_size)]
    mesh = DeviceMesh(shape=(world_size,), axis_names=("model",), devices=global_devices)
    
    # Define layout map
    layout_map = LayoutMap(mesh)
    layout_map[".*dense.*kernel"] = TensorLayout(axes=(None, "model"), device_mesh=mesh)
    layout_map[".*embeddings/embeddings"] = TensorLayout(axes=("model", None), device_mesh=mesh)
    
    # Create the ModelParallel distribution strategy
    distribution = ModelParallel(layout_map=layout_map, batch_dim_name="model")
    
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
        
        # Execute fit
        print(f"Rank {rank} starting fit...")
        x_str = ["Keras Hub is great.", "Distributed training works with ModelParallel!"] * 2
        model.fit(x_str, x_str, epochs=1, batch_size=2)
        
        if rank == 0:
            print("Successfully finished model.fit on OPT-125M with ModelParallel!")

    # Cleanup
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    # Pre-download using a clean subprocess to avoid parent process touching GPU
    import subprocess
    print("Pre-downloading preset...")
    subprocess.run([sys.executable, "-c", "import os; os.environ['CUDA_VISIBLE_DEVICES']=''; os.environ['KERAS_BACKEND']='torch'; import keras_hub; keras_hub.models.OPTCausalLM.from_preset('opt_125m_en', load_weights=False)"])
    
    # Determine world size
    import torch
    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()
    else:
        world_size = 2
        
    # Use 'spawn' to ensure a clean slate for each process
    mp.set_start_method('spawn', force=True)
    try:
        mp.spawn(run_test, args=(world_size,), nprocs=world_size, join=True)
    except Exception as e:
        print(f"Test failed with error: {e}")