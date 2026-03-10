import os
import argparse
import numpy as np
import keras
import keras_hub
import jax
import json

# Set float32 as default (change to float64 if needed for higher precision)
keras.config.set_floatx("float32")

# Set global random seed
keras.utils.set_random_seed(42)

# Set JAX to use GPU if available
if os.environ.get("KERAS_BACKEND") == "jax":
    # JAX handles multi-GPU automatically if visible
    pass

# Set Torch settings
if os.environ.get("KERAS_BACKEND") == "torch":
    import torch
    # Try to avoid MPS issues on Mac
    try:
        if hasattr(torch, "mps"):
            if not hasattr(torch.mps, "is_initialized"):
                torch.mps.is_initialized = lambda: False
            if not hasattr(torch.mps, "set_device"):
                torch.mps.set_device = lambda x: None
    except:
        pass

parser = argparse.ArgumentParser()
parser.add_argument("--strategy", choices=["mp", "dp"], required=True)
parser.add_argument("--steps", type=int, default=1)
parser.add_argument("--epochs", type=int, default=1)
args = parser.parse_args()

def get_model():
    config = {
        "vocabulary_size": 1000,
        "num_layers": 2,
        "num_heads": 2,
        "hidden_dim": 64,
        "intermediate_dim": 128,
        "max_sequence_length": 32,
        "dropout": 0.0,
    }
    model = keras_hub.models.OPTBackbone(**config)
    return model

def get_layout_map(mesh, model):
    from keras.distribution import LayoutMap, TensorLayout
    layout_map = LayoutMap(mesh)
    for var in model.variables:
        path = var.path
        axes = [None] * len(var.shape)
        if "embeddings" in path: axes[-1] = "model"
        elif "feedforward_intermediate_dense/kernel" in path: axes[-1] = "model"
        elif "feedforward_intermediate_dense/bias" in path: axes[0] = "model"
        elif "feedforward_output_dense/kernel" in path: axes[0] = "model"
        elif "feedforward_output_dense/bias" in path: axes[0] = "model"
        elif "layer_norm" in path and ("gamma" in path or "beta" in path): axes[0] = "model"
        elif "self_attention" in path:
            if "/kernel" in path: axes[1 if len(var.shape) == 3 else 0] = "model"
            elif "/bias" in path: axes[0] = "model"
        if "model" not in axes and len(var.shape) > 0: axes[0] = "model"
        layout = TensorLayout(axes=tuple(axes), device_mesh=mesh)
        layout_map[path] = layout
    return layout_map

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
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        keras.distribution.initialize()
    
    from keras.distribution import DeviceMesh, ModelParallel, DataParallel
    
    # Detect devices
    if backend == "jax":
        # JAX simulation or actual GPUs
        devices = jax.local_devices()
        device_names = [str(d) for d in devices]
        world_size = len(devices)
    else: # torch
        if torch.cuda.is_available():
            device_names = [f"cuda:{i}" for i in range(world_size)]
        else:
            device_names = [f"cpu:{i}" for i in range(world_size)]
    
    mesh_shape = (world_size,)
    if args.strategy == "mp":
        mesh = DeviceMesh(shape=mesh_shape, axis_names=("model",), devices=device_names)
        tmp_model = get_model()
        tmp_model({"token_ids": np.zeros((1, 32), dtype="int32"), "padding_mask": np.ones((1, 32), dtype="int32")})
        layout_map = get_layout_map(mesh, tmp_model)
        distribution = ModelParallel(layout_map=layout_map, auto_shard_dataset=False)
    else:
        mesh = DeviceMesh(shape=mesh_shape, axis_names=("batch",), devices=device_names)
        distribution = DataParallel(device_mesh=mesh, auto_shard_dataset=False)

    with distribution.scope():
        model = get_model()
        optimizer = keras.optimizers.Adam(learning_rate=1e-3, epsilon=1e-7)
        model.compile(optimizer=optimizer, loss=keras.losses.MeanSquaredError())
        model.load_weights("initial_weights.weights.h5")
        
        data = np.load("data.npz")
        token_ids = data["token_ids"]
        padding_mask = data["padding_mask"]
        y = data["y"]
        
        global_batch_size = 4
        # Multi-process backend (Torch) manual sharding
        if backend == "torch" and world_size > 1:
            num_samples = token_ids.shape[0]
            indices = np.arange(num_samples)
            rank_indices = []
            samples_per_rank = global_batch_size // world_size
            for i in range(0, num_samples, global_batch_size):
                batch_indices = indices[i : i + global_batch_size]
                if len(batch_indices) == global_batch_size:
                    start = rank * samples_per_rank
                    end = (rank + 1) * samples_per_rank
                    rank_indices.extend(batch_indices[start:end])
            rank_indices = np.array(rank_indices)
            train_x = {"token_ids": token_ids[rank_indices], "padding_mask": padding_mask[rank_indices]}
            train_y = y[rank_indices]
            local_batch_size = samples_per_rank
        else:
            train_x = {"token_ids": token_ids, "padding_mask": padding_mask}
            train_y = y
            local_batch_size = global_batch_size
            
        history_list = []
        def get_manual_loss():
            samples_to_eval = 20
            eval_x = {k: v[:samples_to_eval] for k, v in {"token_ids": token_ids, "padding_mask": padding_mask}.items()}
            out = model(eval_x)
            if isinstance(out, (list, tuple)): out = out[0]
            out_np = keras.ops.convert_to_numpy(out)
            loss = np.mean((out_np - y[:samples_to_eval])**2)
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
            history_path = f"history_{backend}_{args.strategy}.json"
            with open(history_path, "w") as f:
                json.dump({"loss": history_list}, f)
            
            save_path = f"weights_{backend}_{args.strategy}.weights.h5"
            if args.strategy == "dp":
                model.save_weights(save_path)
            else:
                # MP save: gather to plain model
                plain_model = get_model()
                plain_model({"token_ids": np.zeros((1, 32), dtype="int32"), "padding_mask": np.ones((1, 32), dtype="int32")})
                for v_dist, v_plain in zip(model.variables, plain_model.variables):
                    global_tensor = v_dist.value.full_tensor() if hasattr(v_dist.value, "full_tensor") else v_dist
                    v_plain.assign(keras.ops.convert_to_numpy(global_tensor))
                plain_model.save_weights(save_path)
            print(f"Weights saved to {save_path}")

    if backend == "torch":
        import torch.distributed as dist
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    run_training()
