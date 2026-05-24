import os
import sys

# Set backend and FORCE CPU BEFORE importing keras/torch
os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["MPS_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import json
import keras
import torch

# Absolute CPU enforcement: Monkeypatch torch.device to ignore 'mps'
original_device = torch.device
def force_cpu_device(device_type, *args, **kwargs):
    if isinstance(device_type, str) and 'mps' in device_type:
        return original_device('cpu')
    return original_device(device_type, *args, **kwargs)
torch.device = force_cpu_device

# Disable MPS availability globally
torch.backends.mps.is_available = lambda: False
if hasattr(torch.backends, "mps") and hasattr(torch.backends.mps, "is_built"):
    torch.backends.mps.is_built = lambda: False

from shared_utils import get_layout_map, get_data, train_model

def run_torch(rank, world_size):
    # Setup environment for Torch distributed
    os.environ.update({
        "RANK": str(rank),
        "WORLD_SIZE": str(world_size),
        "LOCAL_RANK": str(rank),
        "MASTER_ADDR": "localhost",
        "MASTER_PORT": "29560",
    })
    
    # Re-enforce CPU default in the new process
    torch.set_default_device('cpu')
    
    keras.utils.set_random_seed(42)
    keras.distribution.initialize()
    
    # Create Mesh and ModelParallel distribution
    devices = keras.distribution.list_devices("cpu")[:world_size]
    mesh = keras.distribution.DeviceMesh((world_size,), ["model"], devices)
    dist = keras.distribution.ModelParallel(layout_map=get_layout_map(mesh), batch_dim_name="model")
    
    # Use global data; Keras shards it automatically across processes
    x, y = get_data()
    loss = train_model(dist, x, y, batch_size=4)
    
    if rank == 0:
        with open("results_torch.json", "w") as f:
            json.dump({"final_loss": loss}, f)
            
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    # Ensure spawn starts fresh processes
    torch.multiprocessing.spawn(run_torch, args=(2,), nprocs=2, join=True)
