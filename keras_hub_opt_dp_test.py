import os
import sys
import torch.multiprocessing as mp

# DO NOT import keras or torch at top level to avoid premature CUDA initialization

def run_test(rank, world_size):
    # Set environment variables for this process - MUST BE FIRST
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    os.environ["KERAS_BACKEND"] = "torch"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29507"
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
    from keras.distribution import DataParallel

    # Set device
    torch.cuda.set_device(0)
    
    # Initialize Keras distribution system
    print(f"Rank {rank} initializing distribution...")
    keras.distribution.initialize()
    print(f"Rank {rank} distribution initialized.")
    
    # Get available devices
    devices = keras.distribution.list_devices("gpu")
    print(f"Rank {rank}/{world_size} starting with local devices: {devices}")
    
    # Create DataParallel strategy
    global_devices = [f"gpu:{i}" for i in range(world_size)]
    distribution = DataParallel(devices=global_devices, auto_shard_dataset=False)
    
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
        
        # Generate dummy data
        x = ["Keras Hub is great.", "Distributed training works with PyTorch!"] * 4
        y = x
        
        # Execute fit
        print(f"Rank {rank} starting fit...")
        model.fit(x, y, epochs=1, batch_size=2)
        
        if rank == 0:
            print("Successfully finished model.fit on OPT-125M with DataParallel!")

    # Cleanup
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    # Pre-download using a clean subprocess
    import subprocess
    print("Pre-downloading preset...")
    subprocess.run([sys.executable, "-c", "import os; os.environ['CUDA_VISIBLE_DEVICES']=''; os.environ['KERAS_BACKEND']='torch'; import keras_hub; keras_hub.models.OPTCausalLM.from_preset('opt_125m_en', load_weights=False)"])
    
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
