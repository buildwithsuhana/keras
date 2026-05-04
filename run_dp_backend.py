import os
import sys
import json
import numpy as np

# Set backend immediately
if len(sys.argv) > 1 and sys.argv[1] in ["jax", "torch"]:
    os.environ["KERAS_BACKEND"] = sys.argv[1]

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["KERAS_BACKEND_DEVICE"] = "cpu"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

if os.environ.get("KERAS_BACKEND") == "torch":
    import torch
    if hasattr(torch, "set_default_device"): torch.set_default_device("cpu")
    if hasattr(torch.backends, "mps"):
        torch.backends.mps.is_available = lambda: False

import keras
import keras_hub

def run_training(backend, rank=0):
    keras.config.enable_interactive_logging()
    keras.utils.set_random_seed(42)
    if backend == "torch": keras.distribution.initialize()
    
    world_size = 2
    devices = keras.distribution.list_devices("cpu")
    mesh = keras.distribution.DeviceMesh(
        shape=(world_size,), axis_names=("batch",), 
        devices=devices[:world_size] if len(devices) >= world_size else ["cpu:0"]*world_size
    )
    
    distribution = keras.distribution.DataParallel(
        device_mesh=mesh, auto_shard_dataset=False
    )

    with distribution.scope():
        model = keras_hub.models.OPTBackbone.from_preset("opt_125m_en")
        model.load_weights("initial_weights.weights.h5")
             
        for layer in model._flatten_layers():
            for attr in ["dropout", "dropout_rate", "hidden_dropout_rate", "attention_dropout_rate"]:
                if hasattr(layer, attr):
                    try: setattr(layer, attr, 0.0)
                    except: pass
             
        data = np.load("test_data.npz")
        x_tokens = data["x_tokens"][:2]
        x_mask = data["x_mask"][:2]
        y = data["y"][:2]

        if backend == "torch":
            samples_per_rank = len(y) // world_size
            start, end = rank * samples_per_rank, (rank + 1) * samples_per_rank
            x_tokens, x_mask, y = x_tokens[start:end], x_mask[start:end], y[start:end]

        x = {"token_ids": x_tokens, "padding_mask": x_mask}

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        )

        print(f"[{backend} Rank {rank}] Training...")
        history = model.fit(x, y, epochs=1, verbose=1)
        loss_val = float(history.history["loss"][-1])
        
        print(f"[{backend} Rank {rank}] Loss: {loss_val:.8f}")
        return loss_val

def _torch_worker(rank, world_size, return_dict):
    os.environ["RANK"], os.environ["WORLD_SIZE"] = str(rank), str(world_size)
    os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"] = "localhost", "29507"
    try: return_dict[f"loss_{rank}"] = run_training("torch", rank=rank)
    except Exception as e: print(f"[Torch Rank {rank}] FAILED: {e}")

if __name__ == "__main__":
    backend = sys.argv[1]
    if backend == "jax":
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
        loss = run_training("jax")
    else:
        import torch.multiprocessing as mp
        try: mp.set_start_method('spawn', force=True)
        except: pass
        manager = mp.Manager()
        return_dict = manager.dict()
        mp.spawn(_torch_worker, args=(2, return_dict), nprocs=2, join=True)
        loss = sum(return_dict.values()) / len(return_dict) if return_dict else None
    
    if loss:
        with open(f"loss_{backend}.json", "w") as f: json.dump({"loss": loss}, f)
