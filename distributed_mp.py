import os
import sys

# Set backend BEFORE any imports
if len(sys.argv) > 1:
    os.environ["KERAS_BACKEND"] = sys.argv[1]

# Prevent JAX pre-allocation
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

def run_backend(backend, world_size=2):
    # CRITICAL: Parent process stays tiny. No ML imports here!
    import json
    
    f = f"results_{backend}.json"
    if os.path.exists(f): os.remove(f)
    
    if backend == "jax":
        # JAX worker
        _run_jax(world_size)
    elif backend == "torch":
        import torch
        print(f"Using Torch spawn (Required for GPU safety)")
        torch.multiprocessing.spawn(_run_torch, args=(world_size,), nprocs=world_size, join=True)
        
        # Aggregate results
        import numpy as np # Import only when needed for aggregation
        if os.path.exists("results_torch_rank_0.json"):
            with open("results_torch_rank_0.json", "r") as f:
                results = json.load(f)
            all_peaks = []
            for r in range(world_size):
                fname = f"results_torch_rank_{r}.json"
                if os.path.exists(fname):
                    with open(fname, "r") as f:
                        data = json.load(f)
                        all_peaks.append(data.get("rank_peak_memory", 0))
            if all_peaks:
                results["peak_memory"] = sum(all_peaks)
            with open("results_torch.json", "w") as f: json.dump(results, f, indent=2)

def _run_jax(world_size):
    # Imports happen ONLY inside the worker
    import jax
    import keras
    import keras_hub
    import numpy as np
    import time
    import json
    import psutil
    
    # Simple Memory Tracker inside worker
    process = psutil.Process(os.getpid())
    
    keras.utils.set_random_seed(42)
    devices = keras.distribution.list_devices()
    if len(devices) > world_size: devices = devices[:world_size]
    
    mesh = keras.distribution.DeviceMesh(shape=(world_size,), axis_names=("model",), devices=devices)
    layout_map = keras.distribution.LayoutMap(mesh)
    layout_map[".*token_embedding/embeddings"] = keras.distribution.TensorLayout(("model", None), mesh)
    layout_map[".*position_embedding/embeddings"] = keras.distribution.TensorLayout((None, "model"), mesh)
    layout_map[".*feedforward_intermediate_dense/kernel"] = keras.distribution.TensorLayout((None, "model"), mesh)
    layout_map[".*feedforward_output_dense/kernel"] = keras.distribution.TensorLayout(("model", None), mesh)

    distribution = keras.distribution.ModelParallel(layout_map=layout_map, batch_dim_name="model")
    with distribution.scope():
        model = keras_hub.models.OPTBackbone.from_preset("opt_125m_en", dropout=0.0)
        model.compile(optimizer="adam", loss="mse", jit_compile=False)
        
        x = {"token_ids": np.ones((world_size * 2, 32), dtype="int32"), "padding_mask": np.ones((world_size * 2, 32), dtype="int32")}
        y = np.ones((world_size * 2, 32, 768), dtype="float32")

        # Compilation
        start_comp = time.time()
        model.fit(x, y, epochs=1, steps_per_epoch=1, verbose=0)
        compilation_time = time.time() - start_comp
        
        # Training
        start_train = time.time()
        history = model.fit(x, y, epochs=5, steps_per_epoch=1, verbose=0)
        training_time = time.time() - start_train
        
    peak_mem = process.memory_info().rss / (1024 * 1024)
    
    # Device VRAM
    per_device_mem = {}
    for i, d in enumerate(devices):
        try:
            stats = d.memory_stats()
            per_device_mem[f"Device {i}"] = stats.get('peak_bytes_in_use', 0) / (1024 * 1024)
        except:
            per_device_mem[f"Device {i}"] = 0

    results = {
        "step_1_loss": float(history.history["loss"][0]),
        "step_5_loss": float(history.history["loss"][4]),
        "perplexity": float(np.exp(history.history["loss"][4])),
        "throughput": (world_size * 2 * 5) / training_time,
        "training_time": training_time,
        "compilation_time": compilation_time,
        "peak_memory": peak_mem,
        "per_device_memory": per_device_mem,
    }
    with open("results_jax.json", "w") as f: json.dump(results, f, indent=2)

def _run_torch(rank, world_size):
    # Imports happen ONLY inside the worker
    import torch
    import keras
    import keras_hub
    import numpy as np
    import time
    import json
    import psutil
    import gc

    process = psutil.Process(os.getpid())
    torch.set_num_threads(1)
    
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29560"
    
    keras.utils.set_random_seed(42)
    keras.distribution.initialize()
    
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    devices = keras.distribution.list_devices(device_type)[:world_size]
    mesh = keras.distribution.DeviceMesh(shape=(world_size,), axis_names=("model",), devices=devices)
    layout_map = keras.distribution.LayoutMap(mesh)
    layout_map[".*token_embedding/embeddings"] = keras.distribution.TensorLayout(("model", None), mesh)
    layout_map[".*position_embedding/embeddings"] = keras.distribution.TensorLayout((None, "model"), mesh)
    layout_map[".*feedforward_intermediate_dense/kernel"] = keras.distribution.TensorLayout((None, "model"), mesh)
    layout_map[".*feedforward_output_dense/kernel"] = keras.distribution.TensorLayout(("model", None), mesh)

    distribution = keras.distribution.ModelParallel(layout_map=layout_map, batch_dim_name="model")
    with distribution.scope():
        model = keras_hub.models.OPTBackbone.from_preset("opt_125m_en", dropout=0.0)
        gc.collect()
        if device_type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        model.compile(optimizer="adam", loss="mse", jit_compile=False)
        
        x_full = {"token_ids": np.ones((world_size * 2, 32), dtype="int32"), "padding_mask": np.ones((world_size * 2, 32), dtype="int32")}
        y_full = np.ones((world_size * 2, 32, 768), dtype="float32")
        
        # Local slice for rank
        indices = [rank * 2, rank * 2 + 1]
        x = {k: v[indices] for k, v in x_full.items()}
        y = y_full[indices]
        
        start_comp = time.time()
        model.fit(x, y, batch_size=2, epochs=1, steps_per_epoch=1, verbose=0)
        compilation_time = time.time() - start_comp
        
        start_time = time.time()
        history = model.fit(x, y, batch_size=2, epochs=5, steps_per_epoch=1, verbose=0)
        training_time = time.time() - start_time

    peak_host_mem = process.memory_info().rss / (1024 * 1024)
    device_peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024) if device_type == "cuda" else peak_host_mem

    rank_results = {
        "rank_peak_memory": peak_host_mem,
        "device_peak_memory": device_peak_mem
    }
    
    if rank == 0:
        rank_results.update({
            "step_1_loss": float(history.history["loss"][0]),
            "step_5_loss": float(history.history["loss"][4]),
            "perplexity": float(np.exp(history.history["loss"][4])),
            "throughput": (world_size * 2 * 5) / training_time,
            "training_time": training_time,
            "compilation_time": compilation_time,
        })
    
    with open(f"results_torch_rank_{rank}.json", "w") as f:
        json.dump(rank_results, f, indent=2)
    
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    run_backend(sys.argv[1])
