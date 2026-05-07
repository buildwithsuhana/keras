import os
import sys
import numpy as np
import json
import time

def get_weights_summary(model):
    summary = {}
    target_variables = [
        "token_embedding/embeddings",
        "position_embedding/embeddings",
        "feedforward_intermediate_dense/kernel",
        "feedforward_output_dense/kernel",
    ]
    for v in model.trainable_variables:
        for target in target_variables:
            if target in v.path:
                if v.path not in summary:
                    w = v.numpy()
                    summary[v.path] = {
                        "mean": float(np.mean(w)),
                        "val": w
                    }
    return summary

def run_backend(backend, world_size=2):
    os.environ["KERAS_BACKEND"] = backend
    if backend == "jax":
        num_gpus = 0
        try:
            import torch
            num_gpus = torch.cuda.device_count()
        except:
            pass
        
        if num_gpus < world_size:
            os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={world_size}"
            # Force JAX to CPU if we are simulating to avoid using a single GPU if present
            os.environ["JAX_PLATFORMS"] = "cpu"
        _run_jax(world_size)
    elif backend == "torch":
        import torch
        torch.multiprocessing.spawn(_run_torch, args=(world_size,), nprocs=world_size, join=True)

def _run_jax(world_size):
    import jax
    import keras
    import keras_hub
    keras.utils.set_random_seed(42)
    
    devices = keras.distribution.list_devices()
    if len(devices) > world_size:
        devices = devices[:world_size]
    print(f"Using JAX devices: {devices}")
    
    if len(devices) < world_size:
        raise ValueError(f"Not enough devices found. Expected {world_size}, got {len(devices)}. "
                         f"Check XLA_FLAGS or GPU availability.")

    mesh = keras.distribution.DeviceMesh(shape=(world_size,), axis_names=("model",), devices=devices)
    layout_map = keras.distribution.LayoutMap(mesh)
    
    # Sharding strategy for all layers
    layout_map[".*token_embedding/embeddings"] = keras.distribution.TensorLayout(("model", None), mesh)
    layout_map[".*position_embedding/embeddings"] = keras.distribution.TensorLayout((None, "model"), mesh)
    
    # MHA: Q/K/V shard on heads, Attention Output replicated to avoid Torch view errors
    layout_map[".*self_attention/query/kernel"] = keras.distribution.TensorLayout((None, "model", None), mesh)
    layout_map[".*self_attention/key/kernel"] = keras.distribution.TensorLayout((None, "model", None), mesh)
    layout_map[".*self_attention/value/kernel"] = keras.distribution.TensorLayout((None, "model", None), mesh)
    layout_map[".*self_attention/attention_output/kernel"] = keras.distribution.TensorLayout((None, None, None), mesh)
    
    # MLP: Intermediate shard on output dim, Output shard on input dim
    layout_map[".*feedforward_intermediate_dense/kernel"] = keras.distribution.TensorLayout((None, "model"), mesh)
    layout_map[".*feedforward_output_dense/kernel"] = keras.distribution.TensorLayout(("model", None), mesh)
    
    # Biases
    layout_map[".*self_attention/query/bias"] = keras.distribution.TensorLayout(("model", None), mesh)
    layout_map[".*self_attention/key/bias"] = keras.distribution.TensorLayout(("model", None), mesh)
    layout_map[".*self_attention/value/bias"] = keras.distribution.TensorLayout(("model", None), mesh)
    layout_map[".*self_attention/attention_output/bias"] = keras.distribution.TensorLayout((None,), mesh)
    layout_map[".*feedforward_intermediate_dense/bias"] = keras.distribution.TensorLayout(("model",), mesh)
    layout_map[".*feedforward_output_dense/bias"] = keras.distribution.TensorLayout((None,), mesh)
    
    # LayerNorm (replicated to avoid shape mismatch with normalized_shape)
    layout_map[".*layer_norm/gamma"] = keras.distribution.TensorLayout((None,), mesh)
    layout_map[".*layer_norm/beta"] = keras.distribution.TensorLayout((None,), mesh)

    distribution = keras.distribution.ModelParallel(layout_map=layout_map, batch_dim_name="model", auto_shard_dataset=False)
    with distribution.scope():
        model = keras_hub.models.OPTBackbone.from_preset("opt_125m_en", dropout=0.0)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5), loss="mse")
        
        initial = get_weights_summary(model)
        np.random.seed(42)
        x_full = {
            "token_ids": np.random.randint(0, 50272, (40, 32)).astype("int32"),
            "padding_mask": np.ones((40, 32), dtype="int32")
        }
        y_full = np.random.normal(size=(40, 32, 768)).astype("float32")

        for i in range(10):
            base = i * 4
            x_full["token_ids"][base+2:base+4] = x_full["token_ids"][base:base+2]
            y_full[base+2:base+4] = y_full[base:base+2]

        # Warmup
        model.fit(x_full, y_full, batch_size=4, epochs=1, steps_per_epoch=1, verbose=1, shuffle=False)
        
        start_time = time.time()
        history = model.fit(x_full, y_full, batch_size=4, epochs=5, steps_per_epoch=1, verbose=1, shuffle=False)
        end_time = time.time()
        training_time = end_time - start_time
        
        step_1_loss = float(history.history["loss"][0])
        step_5_loss = float(history.history["loss"][4])
        throughput = (4 * 5) / training_time
        perplexity = float(np.exp(step_5_loss))
        after_step_1 = get_weights_summary(model)

    results = {
        "initial_weights": {k: {"mean": v["mean"]} for k, v in initial.items()},
        "step_1_loss": step_1_loss,
        "step_5_loss": step_5_loss,
        "perplexity": perplexity,
        "throughput": throughput,
        "training_time": training_time,
        "step_1_updates": {path: {"mean": float(np.mean(after_step_1[path]["val"] - initial[path]["val"])), "samples": (after_step_1[path]["val"] - initial[path]["val"]).flatten()[:5].tolist()} for path in initial if path in after_step_1},
    }
    with open("results_jax.json", "w") as f: json.dump(results, f, indent=2)

def _run_torch(rank, world_size):
    import os
    import torch
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29560"
    
    num_gpus = torch.cuda.device_count()
    if num_gpus >= world_size:
        os.environ["KERAS_TORCH_DEVICE"] = "cuda"
        device_type = "cuda"
    else:
        os.environ["KERAS_TORCH_DEVICE"] = "cpu"
        device_type = "cpu"
    
    import keras
    import keras_hub
    import sys
    sys.path.insert(0, os.getcwd())
    keras.utils.set_random_seed(42)
    keras.distribution.initialize()
    
    print(f"[Rank {rank}] Initialized. World size: {world_size}")
    
    devices = keras.distribution.list_devices(device_type)[:world_size]
    print(f"[Rank {rank}] Using devices: {devices}")
    
    mesh = keras.distribution.DeviceMesh(shape=(world_size,), axis_names=("model",), devices=devices)
    layout_map = keras.distribution.LayoutMap(mesh)
    
    # Sharding strategy for all layers
    layout_map[".*token_embedding/embeddings"] = keras.distribution.TensorLayout(("model", None), mesh)
    layout_map[".*position_embedding/embeddings"] = keras.distribution.TensorLayout((None, "model"), mesh)
    
    # MHA: Q/K/V shard on heads, Attention Output replicated to avoid Torch view errors
    layout_map[".*self_attention/query/kernel"] = keras.distribution.TensorLayout((None, "model", None), mesh)
    layout_map[".*self_attention/key/kernel"] = keras.distribution.TensorLayout((None, "model", None), mesh)
    layout_map[".*self_attention/value/kernel"] = keras.distribution.TensorLayout((None, "model", None), mesh)
    layout_map[".*self_attention/attention_output/kernel"] = keras.distribution.TensorLayout((None, None, None), mesh)
    
    # MLP: Intermediate shard on output dim, Output shard on input dim
    layout_map[".*feedforward_intermediate_dense/kernel"] = keras.distribution.TensorLayout((None, "model"), mesh)
    layout_map[".*feedforward_output_dense/kernel"] = keras.distribution.TensorLayout(("model", None), mesh)
    
    # Biases
    layout_map[".*self_attention/query/bias"] = keras.distribution.TensorLayout(("model", None), mesh)
    layout_map[".*self_attention/key/bias"] = keras.distribution.TensorLayout(("model", None), mesh)
    layout_map[".*self_attention/value/bias"] = keras.distribution.TensorLayout(("model", None), mesh)
    layout_map[".*self_attention/attention_output/bias"] = keras.distribution.TensorLayout((None,), mesh)
    layout_map[".*feedforward_intermediate_dense/bias"] = keras.distribution.TensorLayout(("model",), mesh)
    layout_map[".*feedforward_output_dense/bias"] = keras.distribution.TensorLayout((None,), mesh)
    
    # LayerNorm (replicated to avoid shape mismatch with normalized_shape)
    layout_map[".*layer_norm/gamma"] = keras.distribution.TensorLayout((None,), mesh)
    layout_map[".*layer_norm/beta"] = keras.distribution.TensorLayout((None,), mesh)

    distribution = keras.distribution.ModelParallel(layout_map=layout_map, batch_dim_name="model", auto_shard_dataset=False)
    with distribution.scope():
        model = keras_hub.models.OPTBackbone.from_preset("opt_125m_en", dropout=0.0)
        
        if rank == 0:
            print(f"\n[Rank {rank}] Verifying weight sharding:")
            from torch.distributed.tensor import DTensor
            for v in model.trainable_variables:
                val = v.value
                sharding_info = "❌ Not Sharded"
                if isinstance(val, DTensor):
                    sharding_info = "✅ Sharded"
                print(f"    {v.path:<60} | {sharding_info}")

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5), loss="mse")
        
        initial = get_weights_summary(model)
        np.random.seed(42)
        x_full = {
            "token_ids": np.random.randint(0, 50272, (40, 32)).astype("int32"),
            "padding_mask": np.ones((40, 32), dtype="int32")
        }
        y_full = np.random.normal(size=(40, 32, 768)).astype("float32")
        
        for i in range(10):
            base = i * 4
            x_full["token_ids"][base+2:base+4] = x_full["token_ids"][base:base+2]
            y_full[base+2:base+4] = y_full[base:base+2]

        indices = []
        for i in range(10):
             base = i * 4
             if rank == 0: indices.extend([base, base + 1])
             else: indices.extend([base + 2, base + 3])
        x = {k: v[indices] for k, v in x_full.items()}
        y = y_full[indices]
        
        from torch.nn.attention import sdpa_kernel
        with sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            # Warmup
            model.fit(x, y, batch_size=2, epochs=1, steps_per_epoch=1, verbose=1 if rank == 0 else 0, shuffle=False)
            
            start_time = time.time()
            history = model.fit(x, y, batch_size=2, epochs=5, steps_per_epoch=1, verbose=1 if rank == 0 else 0, shuffle=False)
            end_time = time.time()
            training_time = end_time - start_time
            
            after_step_1 = get_weights_summary(model)

            if rank == 0:
                step_1_loss = float(history.history["loss"][0])
                step_5_loss = float(history.history["loss"][4])
                throughput = (4 * 5) / training_time
                perplexity = float(np.exp(step_5_loss))

                print(f"\n[Rank {rank}] Verifying gradient sharding:")
                from torch.distributed.tensor import DTensor
                grad_sharded_count = 0
                for v in model.trainable_variables:
                    if v.value.grad is not None:
                        if isinstance(v.value.grad, DTensor):
                            grad_sharded_count += 1
                print(f"    Total sharded gradients found: {grad_sharded_count}/{len(model.trainable_variables)}")

    if rank == 0:
        results = {
            "initial_weights": {k: {"mean": v["mean"]} for k, v in initial.items()},
            "step_1_loss": step_1_loss,
            "step_5_loss": step_5_loss,
            "perplexity": perplexity,
            "throughput": throughput,
            "training_time": training_time,
            "step_1_updates": {path: {"mean": float(np.mean(after_step_1[path]["val"] - initial[path]["val"])), "samples": (after_step_1[path]["val"] - initial[path]["val"]).flatten()[:5].tolist()} for path in initial if path in after_step_1},
        }
        with open("results_torch.json", "w") as f: json.dump(results, f, indent=2)
    
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    run_backend(sys.argv[1])
