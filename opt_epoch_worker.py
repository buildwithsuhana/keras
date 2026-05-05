import os
import sys
import numpy as np
import keras
import keras_hub
import torch

backend = os.environ.get("KERAS_BACKEND", "jax")
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
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
                torch.cuda.set_device(local_rank)
            dist.init_process_group(backend=dist_backend)
        return dist.get_rank(), dist.get_world_size()
    else:
        import jax
        return jax.process_index(), jax.device_count()

def run_test():
    rank, world_size = setup_dist()
    
    if backend == "torch":
        import torch
        device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
        print(f"[torch Rank {rank}] Training on device: {device}")
    else:
        import jax
        print(f"[jax Process {rank}] Training on {jax.local_device_count()} local devices")
    
    # 1. Setup Model Parallel (2 devices)
    mesh = distribution.DeviceMesh(shape=(world_size,), axis_names=("model",))
    layout_map = distribution.LayoutMap(mesh)
    layout_map[".*token_embedding/embeddings"] = (None, "model")
    layout_map[".*query/kernel"] = ("model", None, None)
    layout_map[".*ffn_inner/kernel"] = (None, "model")
    layout_map[".*ffn_outer/kernel"] = ("model", None)
    
    model_parallel = distribution.ModelParallel(
        layout_map=layout_map, auto_shard_dataset=False
    )
    # Set distribution globally to ensure all internal Keras calls see it
    distribution.set_distribution(model_parallel)
    
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
                "token_ids": distribution_lib.distribute_tensor(
                    torch.as_tensor(dummy_token_ids), dummy_layout
                ),
                "padding_mask": distribution_lib.distribute_tensor(
                    torch.as_tensor(dummy_padding_mask), dummy_layout
                )
            }
            dummy_y = distribution_lib.distribute_tensor(
                torch.as_tensor(dummy_targets), dummy_target_layout
            )
            
            y_pred = model(dummy_inputs)
            model.compute_loss(dummy_inputs, dummy_y, y_pred)
            model.optimizer.build(model.trainable_variables)

            # --- PROVE SHARDING ---
            for v in model.weights:
                if "query/kernel" in v.path:
                    DTensor = distribution_lib._get_dtensor()
                    if DTensor is not None and isinstance(v.value, DTensor):
                        sharding_spec = v.value.placements
                        print(f"[torch Rank {rank}] Weight {v.path} sharding spec: {sharding_spec}")
                    else:
                        print(f"[torch Rank {rank}] Weight {v.path} is NOT a DTensor")
                    break
        else:
            y_pred = model({"token_ids": dummy_token_ids, "padding_mask": dummy_padding_mask})
            model.compute_loss(None, dummy_targets, y_pred)
            
            # --- PROVE JAX SHARDING ---
            if rank == 0:
                for v in model.weights:
                    if "query/kernel" in v.path:
                        print(f"[jax Process 0] Weight {v.path} sharding: {v.value.sharding}")
                        break

    # 2. Weight Sync
    if backend == "jax":
        if rank == 0:
            for v in model.weights:
                val = np.array(v.value)
                np.save(f"opt_epoch_initial_{v.path.replace('/', '_')}.npy", val)
    else:
        for v in model.weights:
            path = f"opt_epoch_initial_{v.path.replace('/', '_')}.npy"
            if os.path.exists(path):
                v.assign(np.load(path))

    # 3. Load fixed data
    token_ids = np.load("opt_token_ids_large.npy")
    padding_mask = np.load("opt_padding_mask_large.npy")
    targets = np.load("opt_targets_large.npy")

    # Local slice for the rank
    samples_per_rank = token_ids.shape[0] // world_size
    start = rank * samples_per_rank
    end = (rank + 1) * samples_per_rank
    
    x_local = {
        "token_ids": token_ids[start:end],
        "padding_mask": padding_mask[start:end]
    }
    y_local = targets[start:end]

    batch_size = 4 // world_size # Total batch size 4
    num_epochs = 10

    # 4. Train with model.fit to show progress bars
    class LossHistory(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if rank == 0:
                epoch_num = epoch + 1
                loss = float(logs.get("loss"))
                np.save(f"{backend}_opt_epoch_{epoch_num}_loss.npy", np.array(loss))

    verbose = 1 if rank == 0 else 0
    model.fit(
        x=x_local,
        y=y_local,
        batch_size=batch_size,
        epochs=num_epochs,
        shuffle=False,
        verbose=verbose,
        callbacks=[LossHistory()]
    )

    if backend == "torch":
        import torch.distributed as dist
        dist.destroy_process_group()

if __name__ == "__main__":
    run_test()