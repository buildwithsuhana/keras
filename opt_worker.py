import os

# Set backend first
backend = os.environ.get("KERAS_BACKEND", "jax")

# Isolate GPUs for each rank and hide them from TF to avoid hangs/conflicts
if backend == "torch" and "LOCAL_RANK" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["LOCAL_RANK"]
    # Force Keras to use the isolated GPU (it will always be index 0 due to isolation)
    os.environ["KERAS_TORCH_DEVICE"] = "cuda:0"

# Prevent TensorFlow from grabbing all GPU memory
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
import numpy as np
import keras
import keras_hub
import torch

keras.utils.set_random_seed(42)
keras.config.disable_traceback_filtering()

from keras import distribution
from keras.src import ops

def setup_dist():
    if backend == "torch":
        import torch.distributed as dist
        if not dist.is_initialized():
            dist_backend = "nccl" if torch.cuda.is_available() else "gloo"
            if dist_backend == "nccl":
                # With CUDA_VISIBLE_DEVICES isolation, we always use device 0
                torch.cuda.set_device(0)
            dist.init_process_group(backend=dist_backend)
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
                import jax
                return np.array(val.addressable_data(rank))
            return np.array(val)
    return None

def run_test():
    rank, world_size = setup_dist()
    
    # 1. Setup Model Parallel
    mesh = distribution.DeviceMesh(shape=(world_size,), axis_names=("model",))
    layout_map = distribution.LayoutMap(mesh)
    layout_map[".*token_embedding/embeddings"] = (None, "model")
    layout_map[".*query/kernel"] = ("model", None, None)
    layout_map[".*ffn_inner/kernel"] = (None, "model")
    layout_map[".*ffn_outer/kernel"] = ("model", None)
    
    model_parallel = distribution.ModelParallel(layout_map=layout_map)
    # Set distribution globally to ensure all internal Keras calls see it
    distribution.set_distribution(model_parallel)
    
    with model_parallel.scope():
        model = keras_hub.models.OPTBackbone(
            vocabulary_size=50272,
            num_layers=12, 
            num_heads=12,
            hidden_dim=768,
            intermediate_dim=3072,
            max_sequence_length=2048,
        )
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5, epsilon=1e-7),
            loss="mse" 
        )

        # Build model eagerly with distributed inputs to avoid symbolic build/meta tensor issues
        if backend == "torch":
            from keras.src.backend.torch import distribution_lib
            data_layout = distribution.TensorLayout(("model", None), mesh)
            dummy_ids = torch.zeros((world_size, 32), dtype=torch.int32)
            dummy_mask = torch.ones((world_size, 32), dtype=torch.int32)
            x_dummy = {
                "token_ids": distribution_lib.distribute_tensor(dummy_ids, data_layout),
                "padding_mask": distribution_lib.distribute_tensor(dummy_mask, data_layout)
            }
            # This triggers eager build
            _ = model(x_dummy)

    # 2. Weight Sync
    if backend == "jax":
        if rank == 0:
            for v in model.weights:
                np.save(f"opt_initial_{v.path.replace('/', '_')}.npy", np.array(v.value))
    else:
        # Wait for JAX rank 0 to save files if they don't exist
        for v in model.weights:
            path = f"opt_initial_{v.path.replace('/', '_')}.npy"
            if os.path.exists(path):
                v.assign(np.load(path))

    # 3. Load fixed data
    token_ids = np.load("opt_token_ids.npy")
    padding_mask = np.load("opt_padding_mask.npy")
    targets = np.load("opt_targets.npy")

    # Local slice for the rank
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
            data_layout = distribution.TensorLayout(("model", None), mesh)
            x_torch = {
                "token_ids": distribution_lib.distribute_tensor(
                    torch.as_tensor(x_local["token_ids"]), data_layout
                ),
                "padding_mask": distribution_lib.distribute_tensor(
                    torch.as_tensor(x_local["padding_mask"]), data_layout
                )
            }
            y_torch = distribution_lib.distribute_tensor(
                torch.as_tensor(y_local), data_layout
            )
            
            logs = model.train_on_batch(x_torch, y_torch)
        else:
            logs = model.train_on_batch(x_local, y_local)

        loss = logs["loss"] if isinstance(logs, dict) else logs
        
        # 5. Save shards and losses
        if backend == "jax":
            import jax
            if rank == 0:
                np.save(f"jax_opt_s{step}_loss.npy", np.array(loss))
            # Save shard from each device (simulated here since we run one process for all JAX devices)
            for i in range(jax.device_count()):
                w_shard = get_weight_slice(model, i)
                np.save(f"jax_opt_s{step}_rank{i}.npy", w_shard)
        else:
            if rank == 0:
                np.save(f"torch_opt_s{step}_loss.npy", np.array(loss.cpu() if hasattr(loss, "cpu") else loss))
            w = get_weight_slice(model, rank)
            np.save(f"torch_opt_s{step}_rank{rank}.npy", w)

    if backend == "torch":
        import torch.distributed as dist
        dist.destroy_process_group()

if __name__ == "__main__":
    run_test()
