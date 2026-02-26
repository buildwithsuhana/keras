import os
import sys
import numpy as np
import keras
import keras_hub

# Force CPU for absolute parity on Mac
os.environ["KERAS_DEVICE"] = "cpu"
os.environ["KERAS_TORCH_DEVICE"] = "cpu"

backend = os.environ.get("KERAS_BACKEND", "jax")
keras.utils.set_random_seed(42)
keras.config.disable_traceback_filtering()

from keras import distribution
from keras.src import ops

def setup_dist():
    if backend == "torch":
        import torch.distributed as dist
        if not dist.is_initialized():
            dist.init_process_group(backend="gloo")
        return dist.get_rank(), dist.get_world_size()
    else:
        import jax
        return jax.process_index(), jax.device_count()

def get_weight_slice(model, rank):
    for v in model.weights:
        if "query/kernel" in v.path:
            val = v.value
            if hasattr(val, "to_local"): # Torch DTensor
                return val.to_local().detach().cpu().numpy()
            elif hasattr(val, "addressable_data"): # JAX sharded array
                return np.array(val.addressable_data(rank))
            return np.array(val)
    return None

def run_test():
    rank, world_size = setup_dist()
    
    # 1. Setup Model Parallel (2 devices)
    mesh = distribution.DeviceMesh(shape=(world_size,), axis_names=("model",))
    layout_map = distribution.LayoutMap(mesh)
    layout_map[".*token_embedding/embeddings"] = (None, "model")
    layout_map[".*query/kernel"] = ("model", None, None)
    layout_map[".*ffn_inner/kernel"] = (None, "model")
    layout_map[".*ffn_outer/kernel"] = ("model", None)
    
    model_parallel = distribution.ModelParallel(layout_map=layout_map)
    
    with model_parallel.scope():
        backbone = keras_hub.models.OPTBackbone(
            vocabulary_size=50272,
            num_layers=12, 
            num_heads=12,
            hidden_dim=768,
            intermediate_dim=3072,
            max_sequence_length=2048,
            dropout=0.0,
        )
        model = backbone
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5, epsilon=1e-7),
            loss="mse" 
        )

    # 2. Weight Sync
    if backend == "jax":
        if rank == 0:
            for v in model.weights:
                np.save(f"opt_initial_{v.path.replace('/', '_')}.npy", np.array(v.value))
    else:
        for v in model.weights:
            path = f"opt_initial_{v.path.replace('/', '_')}.npy"
            if os.path.exists(path):
                v.assign(np.load(path))

    # 3. Load fixed data
    token_ids = np.load("opt_token_ids.npy")
    padding_mask = np.load("opt_padding_mask.npy")
    targets = np.load("opt_targets.npy")

    start = rank * (token_ids.shape[0] // world_size)
    end = (rank + 1) * (token_ids.shape[0] // world_size)
    
    x_local = {
        "token_ids": token_ids[start:end],
        "padding_mask": padding_mask[start:end]
    }
    y_local = targets[start:end]

    # 4. Train 10 steps
    for step in range(1, 11):
        if backend == "torch":
            from keras.src.backend.torch import distribution_lib
            x_torch = {
                "token_ids": distribution_lib.distribute_tensor(
                    x_local["token_ids"], model_parallel.get_data_layout(x_local["token_ids"].shape)
                ),
                "padding_mask": distribution_lib.distribute_tensor(
                    x_local["padding_mask"], model_parallel.get_data_layout(x_local["padding_mask"].shape)
                )
            }
            y_torch = distribution_lib.distribute_tensor(
                y_local, model_parallel.get_data_layout(y_local.shape)
            )
            
            logs = model.train_on_batch(x_torch, y_torch)
        else:
            logs = model.train_on_batch(x_local, y_local)

        loss = logs
        if isinstance(logs, dict):
            loss = logs["loss"]
        
        # 5. Save shards and losses
        if backend == "jax":
            import jax
            if rank == 0:
                np.save(f"jax_opt_s{step}_loss.npy", np.array(loss))
            for i in range(jax.local_device_count()):
                w_shard = get_weight_slice(model, i)
                np.save(f"jax_opt_s{step}_rank{i}.npy", w_shard)
        else:
            if rank == 0:
                np.save(f"torch_opt_s{step}_loss.npy", np.array(loss))
            w = get_weight_slice(model, rank)
            np.save(f"torch_opt_s{step}_rank{rank}.npy", w)

    if backend == "torch":
        import torch.distributed as dist
        dist.destroy_process_group()

if __name__ == "__main__":
    run_test()
