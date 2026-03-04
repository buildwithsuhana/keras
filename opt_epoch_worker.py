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

    num_samples = token_ids.shape[0]
    batch_size = 4
    num_steps = num_samples // batch_size
    num_epochs = 10

    # 4. Train 10 epochs
    for epoch in range(1, num_epochs + 1):
        epoch_losses = []
        for step in range(num_steps):
            start = step * batch_size
            end = (step + 1) * batch_size
            
            x_batch = {
                "token_ids": token_ids[start:end],
                "padding_mask": padding_mask[start:end]
            }
            y_batch = targets[start:end]

            if backend == "torch":
                with model_parallel.scope():
                    from keras.src.backend.torch import distribution_lib
                    x_torch = {
                        "token_ids": distribution_lib.distribute_tensor(
                            x_batch["token_ids"], model_parallel.get_data_layout(x_batch["token_ids"].shape)
                        ),
                        "padding_mask": distribution_lib.distribute_tensor(
                            x_batch["padding_mask"], model_parallel.get_data_layout(x_batch["padding_mask"].shape)
                        )
                    }
                    y_torch = distribution_lib.distribute_tensor(
                        y_batch, model_parallel.get_data_layout(y_batch.shape)
                    )
                    logs = model.train_on_batch(x_torch, y_torch)
            else:
                logs = model.train_on_batch(x_batch, y_batch)

            loss = logs["loss"] if isinstance(logs, dict) else logs
            epoch_losses.append(float(np.array(loss)))
        
        avg_epoch_loss = np.mean(epoch_losses)
        if rank == 0:
            np.save(f"{backend}_opt_epoch_{epoch}_loss.npy", np.array(avg_epoch_loss))
            print(f"Epoch {epoch} {backend} Loss: {avg_epoch_loss:.6f}")

    if backend == "torch":
        import torch.distributed as dist
        dist.destroy_process_group()

if __name__ == "__main__":
    run_test()
