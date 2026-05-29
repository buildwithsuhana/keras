import os
os.environ["KERAS_BACKEND"] = "torch"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["KERAS_TORCH_DEVICE"] = "cpu"

import sys
import numpy as np
import torch
import keras
from torch.utils.data import Dataset, DataLoader

def get_layout_map(mesh):
    layout_map = keras.distribution.LayoutMap(mesh)
    layout_map[".*dense/kernel"] = keras.distribution.TensorLayout((None, "model"), mesh)
    layout_map[".*dense/bias"] = keras.distribution.TensorLayout(("model",), mesh)
    return layout_map

def _run_torch(rank, world_size, port):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    os.environ["GLOO_SOCKET_IFNAME"] = "lo0"
    os.environ["TP_SOCKET_IFNAME"] = "lo0"

    import keras
    keras.utils.set_random_seed(42)
    
    print(f"[Rank {rank}] initializing keras.distribution...", flush=True)
    keras.distribution.initialize(num_processes=world_size, process_id=rank)
    print(f"[Rank {rank}] initialization done", flush=True)

    devices = keras.distribution.list_devices("cpu")[:world_size]
    print(f"[Rank {rank}] Using devices: {devices}", flush=True)
    mesh = keras.distribution.DeviceMesh(shape=(2, 2), axis_names=("batch", "model"), devices=devices)
    layout_map = get_layout_map(mesh)
    distribution = keras.distribution.ModelParallel(
        layout_map=layout_map, 
        batch_dim_name="batch", 
        auto_shard_dataset=False
    )
    
    with distribution.scope():
        print(f"[Rank {rank}] building model...", flush=True)
        model = keras.Sequential([
            keras.layers.Dense(16, activation="relu", input_shape=(8,)),
            keras.layers.Dense(8)
        ])
        print(f"[Rank {rank}] model built", flush=True)
        
        model.compile(optimizer="adam", loss="mse")
        print(f"[Rank {rank}] model compiled", flush=True)

        num_samples = 32
        x = np.random.normal(size=(num_samples, 8)).astype("float32")
        y = np.random.normal(size=(num_samples, 8)).astype("float32")
        
        # Manual sharding for DataParallel dimension (batch)
        num_shards = 2
        shard_id = rank // 2
        samples_per_shard = num_samples // num_shards
        x_shard = x[shard_id * samples_per_shard : (shard_id + 1) * samples_per_shard]
        y_shard = y[shard_id * samples_per_shard : (shard_id + 1) * samples_per_shard]

        print(f"[Rank {rank}] starting fit with shard {shard_id}...", flush=True)
        model.fit(x_shard, y_shard, batch_size=4, epochs=1, verbose=1 if rank == 0 else 0)
        print(f"[Rank {rank}] fit done", flush=True)

if __name__ == "__main__":
    world_size = 4
    port = "29519"
    import torch.multiprocessing as mp
    mp.spawn(_run_torch, args=(world_size, port), nprocs=world_size, join=True)
