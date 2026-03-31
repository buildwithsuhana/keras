import os
import sys
import torch.multiprocessing as mp

# DO NOT import keras or torch at top level to avoid premature CUDA initialization

def run_test(rank, world_size):
    # Set environment variables for this process - MUST BE FIRST
    os.environ["KERAS_BACKEND"] = "torch"
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    
    # Now safe to import
    import torch
    import keras
    import keras_hub
    from keras.distribution import DeviceMesh, LayoutMap, ModelParallel, TensorLayout
    from keras.src.backend.torch import distribution_lib as torch_dist_lib
    from keras.src import tree

    # Set device
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    
    # Initialize Keras distribution system
    print(f"Rank {rank} initializing distribution (local_rank: {local_rank})...")
    keras.distribution.initialize()
    if torch.distributed.is_initialized():
        print(f"Rank {rank} process group initialized. Backend: {torch.distributed.get_backend()}")
        torch.distributed.barrier()
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
    distribution = ModelParallel(layout_map=layout_map, batch_dim_name="model", auto_shard_dataset=False)
    
    print(f"Rank {rank} entering distribution scope...")
    with distribution.scope():
        # Load the OPT 125M model from Keras Hub
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        print(f"Rank {rank} loading model...")
        model = keras_hub.models.OPTCausalLM.from_preset("opt_125m_en", load_weights=False)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        print(f"Rank {rank} model loaded.")
        
        # Compile the model
        print(f"Rank {rank} compiling model...")
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            weighted_metrics=["accuracy"]
        )
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        print(f"Rank {rank} model compiled.")
        
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
            
        sharded_inputs = tree.map_structure(shard_input, inputs)
        
        # Forward pass to build
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
    # Detect if we're already run by torchrun
    if "RANK" in os.environ:
        # Already spawned by torchrun
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        print(f"Detected torchrun environment: RANK={rank}, WORLD_SIZE={world_size}")
        run_test(rank, world_size)
    else:
        # Pre-download using a clean subprocess to avoid parent process touching GPU
        import subprocess
        print("Pre-downloading preset and checking GPU count...")
        # Get world size without initializing CUDA in parent
        world_size_cmd = "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 2)"
        try:
            result = subprocess.run(
                [sys.executable, "-c", f"import os; os.environ['CUDA_VISIBLE_DEVICES']=''; {world_size_cmd}"],
                capture_output=True, text=True, check=True
            )
            world_size = int(result.stdout.strip())
        except Exception:
            world_size = 2

        # Pre-download preset
        subprocess.run([sys.executable, "-c", "import os; os.environ['CUDA_VISIBLE_DEVICES']=''; os.environ['KERAS_BACKEND']='torch'; import keras_hub; keras_hub.models.OPTCausalLM.from_preset('opt_125m_en', load_weights=False)"])

        print(f"Starting distributed test with world_size={world_size}...")

        # Use 'spawn' to ensure a clean slate for each process
        mp.set_start_method('spawn', force=True)
        try:
            # Set Master address and port in parent so children inherit or can see them
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "29515"
            mp.spawn(run_test, args=(world_size,), nprocs=world_size, join=True)
        except Exception as e:
            print(f"Test failed with error: {e}")