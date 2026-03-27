import os
import sys
import torch.multiprocessing as mp

# DO NOT import keras or torch at top level to avoid premature CUDA initialization

def run_test(rank, world_size):
    import torch
    # Set environment variables for this process - MUST BE FIRST
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    os.environ["KERAS_BACKEND"] = "torch"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29514"
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    
    # Now safe to import
    import keras
    import keras_hub
    from keras.distribution import DataParallel

    # Set device
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    
    # Initialize Keras distribution system
    print(f"Rank {rank} initializing distribution...")
    keras.distribution.initialize()
    print(f"Rank {rank} distribution initialized.")
    
    # Get available devices
    device_type = "gpu" if torch.cuda.is_available() else "cpu"
    global_devices = [f"{device_type}:{i}" for i in range(world_size)]
    
    # Create DataParallel distribution strategy
    distribution = DataParallel(devices=global_devices, auto_shard_dataset=False)
    
    # Load the OPT 125M model from Keras Hub
    print(f"Rank {rank} loading model...")
    model = keras_hub.models.OPTCausalLM.from_preset("opt_125m_en")
    print(f"Rank {rank} model loaded.")
    
    with distribution.scope():
        # Compile the model
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            weighted_metrics=["accuracy"]
        )
        
        # Create simple dummy data for test (strings → tokenized)
        x_str = ["Keras Hub is great.", "Distributed training works with DataParallel!"] * 4
        y_str = x_str[:]  # Dummy targets
        
        # Execute fit
        print(f"Rank {rank} starting fit...")
        model.fit(x_str, y_str, epochs=1, batch_size=4)
        
        if rank == 0:
            print("Successfully finished model.fit on OPT-125M with DataParallel!")

    # Cleanup
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    # Pre-download using a clean subprocess
    import subprocess
    print("Pre-downloading preset...")
    subprocess.run([sys.executable, "-c", "import os; os.environ['CUDA_VISIBLE_DEVICES']=''; os.environ['KERAS_BACKEND']='torch'; import keras_hub; keras_hub.models.OPTCausalLM.from_preset('opt_125m_en')"])
    
    # Determine world size
    import torch
    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()
        if world_size < 2:
            world_size = 2  # CPU fallback
    else:
        world_size = 2
        
    # Use 'spawn' for clean slate
    mp.set_start_method('spawn', force=True)
    try:
        mp.spawn(run_test, args=(world_size,), nprocs=world_size, join=True)
        print("✅ DataParallel test PASSED!")
    except Exception as e:
        print(f"❌ DataParallel test FAILED: {e}")

