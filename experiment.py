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
        os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={world_size}"
        _run_jax(world_size)
    elif backend == "torch":
        import torch
        torch.multiprocessing.spawn(_run_torch, args=(world_size,), nprocs=world_size, join=True)

def _run_jax(world_size):
    import jax
    import keras
    import keras_hub
    keras.utils.set_random_seed(42)
    
    mesh = keras.distribution.DeviceMesh(shape=(world_size,), axis_names=("model",), devices=keras.distribution.list_devices()[:world_size])
    layout_map = keras.distribution.LayoutMap(mesh)
    
    layout_map[".*token_embedding/embeddings"] = keras.distribution.TensorLayout(("model", None), mesh)
    layout_map[".*position_embedding/embeddings"] = keras.distribution.TensorLayout((None, "model"), mesh)
    layout_map[".*feedforward_intermediate_dense/kernel"] = keras.distribution.TensorLayout((None, "model"), mesh)
    layout_map[".*feedforward_intermediate_dense/bias"] = keras.distribution.TensorLayout(("model",), mesh)
    layout_map[".*feedforward_output_dense/kernel"] = keras.distribution.TensorLayout(("model", None), mesh)
    layout_map[".*feedforward_output_dense/bias"] = keras.distribution.TensorLayout((None,), mesh)

    distribution = keras.distribution.ModelParallel(layout_map=layout_map, batch_dim_name="model", auto_shard_dataset=False)
    with distribution.scope():
        model = keras_hub.models.OPTBackbone.from_preset("opt_125m_en", dropout=0.0)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5, epsilon=1e-7), loss="mse")
        
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
        model.fit(x_full, y_full, batch_size=4, epochs=1, steps_per_epoch=1, verbose=0, shuffle=False)
        
        start_time = time.time()
        history = model.fit(x_full, y_full, batch_size=4, epochs=1, steps_per_epoch=1, verbose=0, shuffle=False)
        end_time = time.time()
        training_time = end_time - start_time
        
        step_1_loss = float(history.history["loss"][0])
        throughput = 4 / training_time
        perplexity = float(np.exp(step_1_loss))
        after_step_1 = get_weights_summary(model)

    results = {
        "initial_weights": {k: {"mean": v["mean"]} for k, v in initial.items()},
        "step_1_loss": step_1_loss,
        "perplexity": perplexity,
        "throughput": throughput,
        "training_time": training_time,
        "step_1_updates": {path: {"mean": float(np.mean(after_step_1[path]["val"] - initial[path]["val"])), "samples": (after_step_1[path]["val"] - initial[path]["val"]).flatten()[:5].tolist()} for path in initial if path in after_step_1},
    }
    with open("results_jax.json", "w") as f: json.dump(results, f, indent=2)

def _run_torch(rank, world_size):
    import os
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29560"
    os.environ["KERAS_TORCH_DEVICE"] = "cpu"
    import keras
    import keras_hub
    import torch
    import sys
    sys.path.insert(0, os.getcwd())
    keras.utils.set_random_seed(42)
    keras.distribution.initialize()
    
    mesh = keras.distribution.DeviceMesh(shape=(world_size,), axis_names=("model",), devices=keras.distribution.list_devices("cpu")[:world_size])
    layout_map = keras.distribution.LayoutMap(mesh)
    
    layout_map[".*token_embedding/embeddings"] = keras.distribution.TensorLayout(("model", None), mesh)
    layout_map[".*position_embedding/embeddings"] = keras.distribution.TensorLayout((None, "model"), mesh)
    layout_map[".*feedforward_intermediate_dense/kernel"] = keras.distribution.TensorLayout((None, "model"), mesh)
    layout_map[".*feedforward_intermediate_dense/bias"] = keras.distribution.TensorLayout(("model",), mesh)
    layout_map[".*feedforward_output_dense/kernel"] = keras.distribution.TensorLayout(("model", None), mesh)
    layout_map[".*feedforward_output_dense/bias"] = keras.distribution.TensorLayout((None,), mesh)
    
    distribution = keras.distribution.ModelParallel(layout_map=layout_map, batch_dim_name="model", auto_shard_dataset=False)
    with distribution.scope():
        model = keras_hub.models.OPTBackbone.from_preset("opt_125m_en", dropout=0.0)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5, epsilon=1e-7), loss="mse")
        
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
            model.fit(x, y, batch_size=2, epochs=1, steps_per_epoch=1, verbose=0, shuffle=False)
            
            start_time = time.time()
            history = model.fit(x, y, batch_size=2, epochs=1, steps_per_epoch=1, verbose=0, shuffle=False)
            end_time = time.time()
            training_time = end_time - start_time
            
            step_1_loss = float(history.history["loss"][0])
            throughput = 4 / training_time
            perplexity = float(np.exp(step_1_loss))
            after_step_1 = get_weights_summary(model)

    if rank == 0:
        results = {
            "initial_weights": {k: {"mean": v["mean"]} for k, v in initial.items()},
            "step_1_loss": step_1_loss,
            "perplexity": perplexity,
            "throughput": throughput,
            "training_time": training_time,
            "step_1_updates": {path: {"mean": float(np.mean(after_step_1[path]["val"] - initial[path]["val"])), "samples": (after_step_1[path]["val"] - initial[path]["val"]).flatten()[:5].tolist()} for path in initial if path in after_step_1},
        }
        with open("results_torch.json", "w") as f: json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_backend(sys.argv[1])
