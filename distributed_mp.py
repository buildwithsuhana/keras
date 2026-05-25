import os
import sys

# Set backend BEFORE importing Keras
if len(sys.argv) > 1:
    os.environ["KERAS_BACKEND"] = sys.argv[1]

import numpy as np
import json
import time
import psutil
import threading
import contextlib
import gc

# Pre-import libraries
import torch
import keras
import keras_hub

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
    if backend == "jax":
        num_gpus = torch.cuda.device_count()
        if num_gpus < world_size:
            os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={world_size}"
            os.environ["JAX_PLATFORMS"] = "cpu"
        _run_jax(world_size)
    elif backend == "torch":
        ctx = torch.multiprocessing.get_context("fork")
        processes = []
        for rank in range(world_size):
            p = ctx.Process(target=_run_torch, args=(rank, world_size))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
            
        # Collect and merge results from rank files
        if os.path.exists("results_torch_rank_0.json"):
            with open("results_torch_rank_0.json", "r") as f:
                results = json.load(f)
            
            # Find global peak memory across all ranks
            all_peaks = []
            for r in range(world_size):
                fname = f"results_torch_rank_{r}.json"
                if os.path.exists(fname):
                    with open(fname, "r") as f:
                        data = json.load(f)
                        if "rank_peak_memory" in data:
                            all_peaks.append(data["rank_peak_memory"])
            
            if all_peaks:
                results["peak_memory"] = max(all_peaks)
                
            with open("results_torch.json", "w") as f: json.dump(results, f, indent=2)

def _run_jax(world_size):
    keras.utils.set_random_seed(42)
    tracker = MemoryTracker()
    tracker.start()

    devices = keras.distribution.list_devices()
    if len(devices) > world_size:
        devices = devices[:world_size]
    print(f"Using JAX devices: {devices}")
    
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

    if world_size > 1:
        distribution = keras.distribution.ModelParallel(layout_map=layout_map, batch_dim_name="model", auto_shard_dataset=False)
        scope = distribution.scope()
    else:
        scope = contextlib.nullcontext()

    with scope:
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

        start_comp = time.time()
        model.fit(x_full, y_full, batch_size=4, epochs=1, steps_per_epoch=1, verbose=1, shuffle=False)
        compilation_time = time.time() - start_comp
        
        start_time = time.time()
        history = model.fit(x_full, y_full, batch_size=4, epochs=5, steps_per_epoch=1, verbose=1, shuffle=False)
        end_time = time.time()
        training_time = end_time - start_time
        
        step_1_loss = float(history.history["loss"][0])
        step_5_loss = float(history.history["loss"][4])
        throughput = (4 * 5) / training_time
        perplexity = float(np.exp(step_5_loss))
        after_step_1 = get_weights_summary(model)

    peak_memory = tracker.stop()

    results = {
        "initial_weights": {k: {"mean": v["mean"]} for k, v in initial.items()},
        "step_1_loss": step_1_loss,
        "step_5_loss": step_5_loss,
        "perplexity": perplexity,
        "throughput": throughput,
        "training_time": training_time,
        "compilation_time": compilation_time,
        "peak_memory": peak_memory,
        "step_1_updates": {path: {"mean": float(np.mean(after_step_1[path]["val"] - initial[path]["val"])), "samples": (after_step_1[path]["val"] - initial[path]["val"]).flatten()[:5].tolist()} for path in initial if path in after_step_1},
    }
    with open("results_jax.json", "w") as f: json.dump(results, f, indent=2)

def _run_torch(rank, world_size):
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

    if world_size > 1:
        distribution = keras.distribution.ModelParallel(layout_map=layout_map, batch_dim_name="model", auto_shard_dataset=False)
        scope = distribution.scope()
    else:
        scope = contextlib.nullcontext()

    with scope:
        model = keras_hub.models.OPTBackbone.from_preset("opt_125m_en", dropout=0.0)
        gc.collect()
        if device_type == "cuda":
            torch.cuda.empty_cache()

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

    peak_memory = tracker.stop()
    
    # Save results to a rank-specific file
    rank_results = {
        "rank_peak_memory": peak_memory
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
    
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    backend = sys.argv[1]
    world_size = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    run_backend(backend, world_size)
