import os
import sys

# CRITICAL: Set backend BEFORE any Keras/Torch imports
if len(sys.argv) > 1:
    os.environ["KERAS_BACKEND"] = sys.argv[1]

# Prevent JAX from pre-allocating all VRAM to avoid OOM in multi-backend scripts
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import json
import time
import psutil
import threading
import contextlib
import gc
import traceback

class MemoryTracker:
    def __init__(self):
        self.peak_memory = 0
        self.running = False
        self.thread = None

    def _track(self):
        process = psutil.Process(os.getpid())
        while self.running:
            try:
                mem = process.memory_info().rss
                if mem > self.peak_memory:
                    self.peak_memory = mem
            except:
                break
            time.sleep(0.1)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._track, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        return self.peak_memory / (1024 * 1024) # MB

def get_weights_summary(model):
    summary = {}
    target_variables = [
        "token_embedding/embeddings",
        "position_embedding/embeddings",
        "feedforward_intermediate_dense/kernel",
        "feedforward_output_dense/kernel",
    ]
    for v in model.trainable_variables:
        path = str(v.path)
        for target in target_variables:
            if target in path:
                if path not in summary:
                    w = v.numpy()
                    summary[path] = {
                        "mean": float(np.mean(w)),
                        "val": w
                    }
    return summary

def run_backend(backend, world_size=2):
    print(f"\n--- Starting {backend.upper()} Run (world_size={world_size}) ---")
    
    # Only clean up this specific backend's results
    f = f"results_{backend}.json"
    if os.path.exists(f): os.remove(f)
    
    if backend == "torch":
        for r in range(world_size):
            rank_f = f"results_torch_rank_{r}.json"
            if os.path.exists(rank_f): os.remove(rank_f)

    if backend == "jax":
        import torch
        num_gpus = torch.cuda.device_count() if hasattr(torch.cuda, "device_count") else 0
        if num_gpus < world_size:
            os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={world_size}"
            os.environ["JAX_PLATFORMS"] = "cpu"
        _run_jax(world_size)
    elif backend == "torch":
        import torch
        # Always use spawn for consistency and safety on both CPU/GPU
        print(f"Using Torch start method: spawn")
        try:
            torch.multiprocessing.spawn(_run_torch, args=(world_size,), nprocs=world_size, join=True)
        except Exception as e:
            print(f"Torch Spawn failed: {e}")
            traceback.print_exc()
            return

        # Aggregate results
        if os.path.exists("results_torch_rank_0.json"):
            with open("results_torch_rank_0.json", "r") as f:
                results = json.load(f)
            
            per_device_mem = {}
            total_host_mem = 0
            for r in range(world_size):
                fname = f"results_torch_rank_{r}.json"
                if os.path.exists(fname):
                    with open(fname, "r") as f:
                        data = json.load(f)
                        per_device_mem[f"Device {r}"] = data.get("device_peak_memory", 0)
                        total_host_mem += data.get("rank_peak_memory", 0)
            
            results["per_device_memory"] = per_device_mem
            results["peak_memory"] = total_host_mem # Total Host RSS
            with open("results_torch.json", "w") as f: json.dump(results, f, indent=2)
            print(f"Final results written to results_torch.json (System Peak Memory: {total_host_mem:.2f} MB)")
        else:
            print("Error: results_torch_rank_0.json not found. Torch run failed to produce results.")

def _run_jax(world_size):
    try:
        import jax
        import keras
        import keras_hub
        keras.utils.set_random_seed(42)
        
        tracker = MemoryTracker()
        tracker.start()

        devices = keras.distribution.list_devices()
        if len(devices) > world_size:
            devices = devices[:world_size]
        print(f"Using JAX devices: {devices}")

        mesh = keras.distribution.DeviceMesh(shape=(world_size,), axis_names=("model",), devices=devices)
        layout_map = keras.distribution.LayoutMap(mesh)
        
        # Sharding
        layout_map[".*token_embedding/embeddings"] = keras.distribution.TensorLayout(("model", None), mesh)
        layout_map[".*position_embedding/embeddings"] = keras.distribution.TensorLayout((None, "model"), mesh)
        layout_map[".*self_attention/query/kernel"] = keras.distribution.TensorLayout((None, "model", None), mesh)
        layout_map[".*self_attention/key/kernel"] = keras.distribution.TensorLayout((None, "model", None), mesh)
        layout_map[".*self_attention/value/kernel"] = keras.distribution.TensorLayout((None, "model", None), mesh)
        layout_map[".*self_attention/attention_output/kernel"] = keras.distribution.TensorLayout(("model", None), mesh)
        layout_map[".*feedforward_intermediate_dense/kernel"] = keras.distribution.TensorLayout((None, "model"), mesh)
        layout_map[".*feedforward_output_dense/kernel"] = keras.distribution.TensorLayout(("model", None), mesh)

        distribution = keras.distribution.ModelParallel(layout_map=layout_map, batch_dim_name="model", auto_shard_dataset=False)
        with distribution.scope():
            model = keras_hub.models.OPTBackbone.from_preset("opt_125m_en", dropout=0.0)
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5), loss="mse", jit_compile=False)
            
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

            print("JAX: Starting Warmup...")
            start_comp = time.time()
            model.fit(x_full, y_full, batch_size=4, epochs=1, steps_per_epoch=1, verbose=1, shuffle=False)
            compilation_time = time.time() - start_comp
            
            print("JAX: Starting Training...")
            start_time = time.time()
            history = model.fit(x_full, y_full, batch_size=4, epochs=5, steps_per_epoch=1, verbose=1, shuffle=False)
            end_time = time.time()
            training_time = end_time - start_time
            
            step_1_loss = float(history.history["loss"][0])
            step_5_loss = float(history.history["loss"][4])
            
            print("JAX: Collecting final weights...")
            after_step_1 = get_weights_summary(model)

        peak_host_memory = tracker.stop()
        
        per_device_mem = {}
        for i, d in enumerate(devices):
            try:
                stats = d.memory_stats()
                mem_mb = stats.get('peak_bytes_in_use', 0) / (1024 * 1024)
                per_device_mem[f"Device {i}"] = mem_mb
            except:
                per_device_mem[f"Device {i}"] = peak_host_memory / world_size

        results = {
            "initial_weights": {k: {"mean": v["mean"]} for k, v in initial.items()},
            "step_1_loss": step_1_loss,
            "step_5_loss": step_5_loss,
            "perplexity": float(np.exp(step_5_loss)),
            "throughput": (4 * 5) / training_time,
            "training_time": training_time,
            "compilation_time": compilation_time,
            "peak_memory": peak_host_memory,
            "per_device_memory": per_device_mem,
            "step_1_updates": {path: {"mean": float(np.mean(after_step_1[path]["val"] - initial[path]["val"])), "samples": (after_step_1[path]["val"] - initial[path]["val"]).flatten()[:5].tolist()} for path in initial if path in after_step_1},
        }
        with open("results_jax.json", "w") as f: json.dump(results, f, indent=2)
        print(f"JAX: Results written to results_jax.json (Peak Memory: {peak_host_memory:.2f} MB)")
    except Exception as e:
        print(f"JAX Run crashed with error: {e}")
        traceback.print_exc()

def _run_torch(rank, world_size):
    try:
        import torch
        import keras
        import keras_hub
        
        torch.set_num_threads(1)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29560"
        
        num_gpus = torch.cuda.device_count()
        device_type = "cuda" if num_gpus >= world_size else "cpu"
        os.environ["KERAS_TORCH_DEVICE"] = device_type
        
        keras.utils.set_random_seed(42)
        keras.distribution.initialize()
        
        tracker = MemoryTracker()
        tracker.start()

        print(f"[Rank {rank}] Initialized on {device_type}")
        
        devices = keras.distribution.list_devices(device_type)[:world_size]
        mesh = keras.distribution.DeviceMesh(shape=(world_size,), axis_names=("model",), devices=devices)
        layout_map = keras.distribution.LayoutMap(mesh)
        
        # Sharding strategy
        layout_map[".*token_embedding/embeddings"] = keras.distribution.TensorLayout(("model", None), mesh)
        layout_map[".*position_embedding/embeddings"] = keras.distribution.TensorLayout((None, "model"), mesh)
        layout_map[".*self_attention/query/kernel"] = keras.distribution.TensorLayout((None, "model", None), mesh)
        layout_map[".*self_attention/key/kernel"] = keras.distribution.TensorLayout((None, "model", None), mesh)
        layout_map[".*self_attention/value/kernel"] = keras.distribution.TensorLayout((None, "model", None), mesh)
        layout_map[".*self_attention/attention_output/kernel"] = keras.distribution.TensorLayout(("model", None), mesh)
        layout_map[".*feedforward_intermediate_dense/kernel"] = keras.distribution.TensorLayout((None, "model"), mesh)
        layout_map[".*feedforward_output_dense/kernel"] = keras.distribution.TensorLayout(("model", None), mesh)

        distribution = keras.distribution.ModelParallel(layout_map=layout_map, batch_dim_name="model", auto_shard_dataset=False)
        with distribution.scope():
            model = keras_hub.models.OPTBackbone.from_preset("opt_125m_en", dropout=0.0)
            gc.collect()
            if device_type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5), loss="mse", jit_compile=False)
            
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
            
            start_comp = time.time()
            model.fit(x, y, batch_size=2, epochs=1, steps_per_epoch=1, verbose=1 if rank == 0 else 0, shuffle=False)
            compilation_time = time.time() - start_comp
            
            start_time = time.time()
            history = model.fit(x, y, batch_size=2, epochs=5, steps_per_epoch=1, verbose=1 if rank == 0 else 0, shuffle=False)
            end_time = time.time()
            training_time = end_time - start_time
            
            after_step_1 = get_weights_summary(model)

        peak_host_memory = tracker.stop()
        
        device_peak_memory = 0
        if device_type == "cuda":
            device_peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
        else:
            device_peak_memory = peak_host_memory

        rank_results = {
            "rank_peak_memory": peak_host_memory,
            "device_peak_memory": device_peak_memory
        }
        
        if rank == 0:
            rank_results.update({
                "initial_weights": {k: {"mean": v["mean"]} for k, v in initial.items()},
                "step_1_loss": float(history.history["loss"][0]),
                "step_5_loss": float(history.history["loss"][4]),
                "perplexity": float(np.exp(history.history["loss"][4])),
                "throughput": (4 * 5) / training_time,
                "training_time": training_time,
                "compilation_time": compilation_time,
                "step_1_updates": {path: {"mean": float(np.mean(after_step_1[path]["val"] - initial[path]["val"])), "samples": (after_step_1[path]["val"] - initial[path]["val"]).flatten()[:5].tolist()} for path in initial if path in after_step_1},
            })
        
        with open(f"results_torch_rank_{rank}.json", "w") as f:
            json.dump(rank_results, f, indent=2)
        print(f"[Rank {rank}] Successfully wrote results_torch_rank_{rank}.json")
        
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
            torch.distributed.destroy_process_group()
    except Exception as e:
        print(f"[Rank {rank}] Run crashed with error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    backend = sys.argv[1]
    world_size = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    run_backend(backend, world_size)
