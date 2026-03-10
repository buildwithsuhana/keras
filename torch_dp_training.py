import os
import torch

# Keras settings
os.environ["KERAS_BACKEND"] = "torch"
os.environ["KERAS_DISTRIBUTION_DEBUG"] = "1"

# Set NCCL environment variables
os.environ["NCCL_DEBUG"] = "WARN"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import keras
import keras_hub
from keras.distribution import DeviceMesh, DataParallel
import torch.distributed as dist
import atexit

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

atexit.register(cleanup)

def train_opt_data_parallel():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")

    keras.distribution.initialize()

    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = dist.get_world_size()

    # Device setup
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        devices = [f"cuda:{i}" for i in range(world_size)]
    else:
        device = "cpu"
        devices = [f"cpu:{i}" for i in range(world_size)]
    
    os.environ["KERAS_TORCH_DEVICE"] = device
    
    mesh_shape = (world_size,)
    mesh = DeviceMesh(shape=mesh_shape, axis_names=("batch",), devices=devices)

    print(f"RANK {rank}: Initializing DataParallel with mesh {mesh}")
    distribution = DataParallel(device_mesh=mesh, auto_shard_dataset=False)

    # Build model INSIDE distribution scope
    print(f"RANK {rank}: Creating and building model within distribution scope...")
    with distribution.scope():
        model = keras_hub.models.OPTBackbone(
            vocabulary_size=1000, 
            num_layers=2, 
            num_heads=2, 
            hidden_dim=64, 
            intermediate_dim=128, 
            max_sequence_length=32, 
            dropout=0.0,
        )
        # Build the model
        model.build({"token_ids": (None, 32), "padding_mask": (None, 32)})
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-4), 
            loss=keras.losses.MeanSquaredError()
        )
        
        # Prepare dummy data
        token_ids = np.random.randint(0, 1000, (16, 32)).astype("int32")
        padding_mask = np.ones((16, 32), dtype="int32")
        y = np.random.randn(16, 32, 64).astype("float32")
        x = {"token_ids": token_ids, "padding_mask": padding_mask}
        
        # Run fit
        print(f"RANK {rank}: Starting model.fit()...")
        history = model.fit(x, y, epochs=1, batch_size=4, verbose=1 if rank == 0 else 0)
        
        print(f"\n✓ model.fit() completed successfully on RANK {rank}!")
        final_loss = float(history.history['loss'][-1])
        print(f"  Final loss on RANK {rank}: {final_loss:.4f}")
            
        # Validation
        print(f"RANK {rank}: Validating distribution state...")
        
        # Check for DDP wrapper
        if hasattr(model, "_ddp_wrapper"):
            print(f"  RANK {rank}: Model has _ddp_wrapper (correct for DDP implementation)")
        else:
            print(f"  RANK {rank}: Model DOES NOT have _ddp_wrapper (multi-process DP expects DDP)")

        is_dtensor = any(hasattr(v.value, 'placements') for v in model.variables)
        if not is_dtensor:
            print(f"  RANK {rank}: Variables are regular Tensors (correct for DDP-based DP)")
        else:
            print(f"  RANK {rank}: Variables are DTensors (unexpected for pure DDP-based DP)")

if __name__ == "__main__":
    train_opt_data_parallel()
