import os
import sys
import numpy as np
import keras
import keras_hub

backend = os.environ.get("KERAS_BACKEND", "jax")
keras.utils.set_random_seed(42)
keras.config.disable_traceback_filtering()

from keras import distribution
from keras.src import ops

def setup_dist():
    if backend == "torch":
        import torch
        import torch.distributed as dist
        if not dist.is_initialized():
            dist_backend = "nccl" if torch.cuda.is_available() else "gloo"
            if dist_backend == "nccl":
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
                torch.cuda.set_device(local_rank)
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
        # Re-set seed inside scope to ensure SeedGenerator state is replicated
        keras.utils.set_random_seed(42)
        
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
            optimizer=keras.optimizers.Adam(learning_rate=1e-4),
            loss="mse" 
        )

        # Build model and loss eagerly with real data to avoid symbolic build issues
        dummy_token_ids = np.zeros((2, 32), dtype="int32")
        dummy_padding_mask = np.ones((2, 32), dtype="int32")
        dummy_targets = np.zeros((2, 32, 768), dtype="float32")
        
        if backend == "torch":
            from keras.src.backend.torch import distribution_lib
            # Use replicated layout for building
            dummy_layout = distribution.TensorLayout((None, None), mesh)
            dummy_target_layout = distribution.TensorLayout((None, None, None), mesh)
            dummy_inputs = {
                "token_ids": distribution_lib.distribute_tensor(dummy_token_ids, dummy_layout),
                "padding_mask": distribution_lib.distribute_tensor(dummy_padding_mask, dummy_layout)
            }
            dummy_y = distribution_lib.distribute_tensor(dummy_targets, dummy_target_layout)
            
            y_pred = model(dummy_inputs)
            model.compute_loss(dummy_inputs, dummy_y, y_pred)
            model.optimizer.build(model.trainable_variables)
        else:
            y_pred = model({"token_ids": dummy_token_ids, "padding_mask": dummy_padding_mask})
            model.compute_loss(None, dummy_targets, y_pred)

    # 2. Weight Sync
    synced_count = 0
    if backend == "jax":
        if rank == 0:
            for v in model.weights:
                val = np.array(v.value)
                np.save(f"opt_initial_{v.path.replace('/', '_')}.npy", val)
                synced_count += 1
            print(f"JAX synced {synced_count} weights")
    else:
        for v in model.weights:
            path = f"opt_initial_{v.path.replace('/', '_')}.npy"
            if os.path.exists(path):
                v.assign(np.load(path))
                synced_count += 1
            else:
                print(f"Sync warning: {v.path} not found")
        if rank == 0:
            print(f"Torch synced {synced_count} weights")

    # 3. Load fixed data
    token_ids = np.load("opt_token_ids.npy")
    padding_mask = np.load("opt_padding_mask.npy")
    targets = np.load("opt_targets.npy")

    # Use full batch on all ranks to ensure identical replicated inputs
    x_local = {
        "token_ids": token_ids,
        "padding_mask": padding_mask
    }
    y_local = targets

    # 4. Train 10 steps
    for step in range(1, 11):
        if backend == "torch":
            with model_parallel.scope():
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
            # JAX: don't use scope during train_on_batch if it causes issues with None sample_weight
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
