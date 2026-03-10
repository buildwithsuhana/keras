import os
import argparse
import numpy as np
import json

# Force float64 for verification
import keras
keras.config.set_floatx("float64")

# Enable JAX 64-bit
os.environ["JAX_ENABLE_X64"] = "True"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import keras_hub

# Set global random seed
keras.utils.set_random_seed(42)

if os.environ.get("KERAS_BACKEND") == "torch":
    import torch
    try:
        if not hasattr(torch, "mps"):
            class DummyMPS:
                def is_initialized(self): return False
                def set_device(self, x): pass
            torch.mps = DummyMPS()
        elif not hasattr(torch.mps, "is_initialized"):
            torch.mps.is_initialized = lambda: False
    except: pass

parser = argparse.ArgumentParser()
parser.add_argument("--strategy", choices=["mp", "dp"], required=True)
parser.add_argument("--steps", type=int, default=1)
parser.add_argument("--epochs", type=int, default=1)
args = parser.parse_args()

def get_model():
    config = {
        "vocabulary_size": 1000, "num_layers": 2, "num_heads": 2,
        "hidden_dim": 64, "intermediate_dim": 128, "max_sequence_length": 32,
        "dropout": 0.0,
    }
    return keras_hub.models.OPTBackbone(**config)

def run_training():
    backend = os.environ.get("KERAS_BACKEND")
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    
    if backend == "torch":
        import torch.distributed as dist
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if torch.cuda.is_available(): torch.cuda.set_device(local_rank)
        keras.distribution.initialize()

    from keras.distribution import DeviceMesh, DataParallel
    num_gpus = 0
    if backend == "torch":
        import torch
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    else:
        import jax
        try: num_gpus = jax.device_count("gpu")
        except: num_gpus = 0
            
    device_names = [f"gpu:{i}" for i in range(world_size)] if num_gpus > 0 else [f"cpu:{i}" for i in range(world_size)]
    
    mesh = DeviceMesh(shape=(world_size,), axis_names=("batch",), devices=device_names)
    distribution = DataParallel(device_mesh=mesh, auto_shard_dataset=False)

    with distribution.scope():
        model = get_model()
        optimizer = keras.optimizers.Adam(learning_rate=1e-3, epsilon=1e-7)
        model.compile(optimizer=optimizer, loss=keras.losses.MeanSquaredError())
        model.load_weights("initial_weights.weights.h5")
        
        data = np.load("data.npz")
        token_ids, padding_mask, y = data["token_ids"], data["padding_mask"], data["y"]
        
        global_batch_size = 4
        if backend == "torch" and world_size > 1:
            num_samples = token_ids.shape[0]
            indices = np.arange(num_samples)
            rank_indices = []
            samples_per_rank = global_batch_size // world_size
            for i in range(0, num_samples, global_batch_size):
                batch_indices = indices[i : i + global_batch_size]
                if len(batch_indices) == global_batch_size:
                    start, end = rank * samples_per_rank, (rank + 1) * samples_per_rank
                    rank_indices.extend(batch_indices[start:end])
            rank_indices = np.array(rank_indices)
            train_x = {"token_ids": token_ids[rank_indices], "padding_mask": padding_mask[rank_indices]}
            train_y, local_batch_size = y[rank_indices], samples_per_rank
        else:
            train_x = {"token_ids": token_ids, "padding_mask": padding_mask}
            train_y, local_batch_size = y, global_batch_size
            
        history_list = []
        def get_manual_loss():
            eval_x = {k: v[:20] for k, v in {"token_ids": token_ids, "padding_mask": padding_mask}.items()}
            out = model(eval_x)
            if isinstance(out, (list, tuple)): out = out[0]
            loss = np.mean((keras.ops.convert_to_numpy(out) - y[:20])**2)
            return float(loss)

        l0 = get_manual_loss()
        history_list.append(l0)
        if rank == 0: print(f"Step 0 global_loss: {l0:.12f}")

        for epoch in range(args.epochs):
            model.fit(train_x, train_y, epochs=1, batch_size=local_batch_size, steps_per_epoch=args.steps, shuffle=False, verbose=0)
            cur_loss = get_manual_loss()
            history_list.append(cur_loss)
            if rank == 0: print(f"Epoch {epoch+1}/{args.epochs} - global_loss: {cur_loss:.12f}")

        if rank == 0:
            with open(f"history_{backend}_{args.strategy}.json", "w") as f:
                json.dump({"loss": history_list}, f)
            model.save_weights(f"weights_{backend}_{args.strategy}.weights.h5")

    if backend == "torch":
        import torch.distributed as dist
        dist.barrier(); dist.destroy_process_group()

if __name__ == "__main__":
    run_training()
