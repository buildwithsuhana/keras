import os
import sys
import json
import numpy as np

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
    mesh = keras.distribution.DeviceMesh(
        shape=(world_size,), axis_names=("model",), 
        devices=keras.distribution.list_devices("cpu")[:world_size]
    )
    
    layout_map = keras.distribution.LayoutMap(mesh)
    layout_map["embeddings/.*embedding/.*"] = keras.distribution.TensorLayout((None, None), mesh)
    layout_map[".*attention.*(query|key|value).*kernel"] = keras.distribution.TensorLayout(("model", None), mesh)

    distribution = keras.distribution.ModelParallel(
        layout_map=layout_map, batch_dim_name="model", auto_shard_dataset=False
    )

    with distribution.scope():
        model = keras_hub.models.OPTBackbone.from_preset("opt_125m_en")
        model.load_weights("initial_weights.weights.h5")
        
        if backend == "torch":
            def custom_train_step(self, data):
                import keras.src.tree as tree
                from keras.src.backend import torch as torch_backend
                (
                    dist_lib,
                    dist,
                    x,
                    y,
                    sample_weight,
                ) = self._unpack_and_distribute_data(data)

                self.zero_grad()
                y_pred = self._forward(dist, x, training=False)
                self._sync_ddp_buffers(dist)

                loss = self._compute_loss(
                    x=x, y=y, y_pred=y_pred, sample_weight=sample_weight, training=False
                )
                self._loss_tracker.update_state(
                    loss,
                    sample_weight=next(
                        i for i in tree.flatten(x) if i is not None
                    ).shape[0],
                )
                return self.compute_metrics(x, y, y_pred, sample_weight=sample_weight)
            
            model.train_step = custom_train_step.__get__(model, model.__class__)
        elif backend == "jax":
            import jax
            from keras.src.trainers.data_adapters import data_adapter_utils
            def custom_train_step(self, state, data):
                (
                    trainable_variables,
                    non_trainable_variables,
                    optimizer_variables,
                    metrics_variables,
                ) = state
                x, y, sample_weight = data_adapter_utils.unpack_x_y_sample_weight(data)
                grad_fn = jax.value_and_grad(
                    self.compute_loss_and_updates, has_aux=True
                )
                (loss, aux), grads = grad_fn(
                    trainable_variables,
                    non_trainable_variables,
                    metrics_variables,
                    x,
                    y,
                    sample_weight,
                    training=False, # FORCE FALSE
                    optimizer_variables=optimizer_variables,
                )
                (unscaled_loss, y_pred, non_trainable_variables, metrics_variables) = (
                    aux
                )

                (
                    trainable_variables,
                    optimizer_variables,
                ) = self.optimizer.stateless_apply(
                    optimizer_variables, grads, trainable_variables
                )

                logs, metrics_variables = self._update_metrics_variables(
                    metrics_variables, unscaled_loss, x, y, y_pred, sample_weight
                )

                state = (
                    trainable_variables,
                    non_trainable_variables,
                    optimizer_variables,
                    metrics_variables,
                )
                return logs, state
            
            model.train_step = custom_train_step.__get__(model, model.__class__)
        
        # For JAX we would need a different approach if we wanted to override it there too
        # But let's first see if Torch with training=False matches JAX
        
        # Disable training to avoid dropout divergence
        model.trainable = False
             
        for layer in model._flatten_layers():
            for attr in ["dropout", "dropout_rate", "hidden_dropout_rate", "attention_dropout_rate"]:
                if hasattr(layer, attr):
                    try: setattr(layer, attr, 0.0)
                    except: pass
        
        data = np.load("test_data.npz")
        x_tokens = data["x_tokens"][:2]
        x_mask = data["x_mask"][:2]
        y = data["y"][:2]
        
        x = {"token_ids": x_tokens, "padding_mask": x_mask}

        # Real training with Adam
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        )

        print(f"[{backend} Rank {rank}] Fitting...")
        
        # Check some weights
        w = model.weights[0]
        w_val = keras.ops.convert_to_numpy(w)
        print(f"[{backend} Rank {rank}] First weight mean: {np.mean(w_val):.8f}")

        history = model.fit(x, y, epochs=1, verbose=1)
        fit_loss = float(history.history["loss"][0])
        return fit_loss
        
        print(f"[{backend} Rank {rank}] Loss: {loss_val:.8f}")
        return loss_val

def _torch_worker(rank, world_size, return_dict):
    os.environ["RANK"], os.environ["WORLD_SIZE"] = str(rank), str(world_size)
    os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"] = "localhost", "29505"
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
