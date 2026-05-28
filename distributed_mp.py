import os
import sys
import numpy as np
import json
import time
import psutil
import threading
import gc

class MemoryTracker:
    def __init__(self):
        self.peak_cpu = 0
        self.base_memory = 0
        self.running = False
        self.thread = None

    def _track(self):
        process = psutil.Process(os.getpid())
        while self.running:
            try:
                mem = process.memory_info().rss
                if mem > self.peak_cpu:
                    self.peak_cpu = mem
            except:
                break
            time.sleep(0.5)

    def start(self):
        gc.collect()
        self.base_memory = psutil.Process(os.getpid()).memory_info().rss
        self.peak_cpu = self.base_memory
        self.running = True
        self.thread = threading.Thread(target=self._track, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        return float(self.peak_cpu)

    def reset(self):
        gc.collect()
        self.base_memory = psutil.Process(os.getpid()).memory_info().rss
        self.peak_cpu = self.base_memory

def find_free_port():
    import socket
    for _ in range(5):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("localhost", 0))
                return s.getsockname()[1]
        except OSError:
            time.sleep(0.1)
            continue
    return 29500 + np.random.randint(0, 1000) # Fallback

def get_layout_map(mesh):
    import keras
    layout_map = keras.distribution.LayoutMap(mesh)
    
    # Sharding strategy for all layers
    layout_map[".*token_embedding/embeddings"] = keras.distribution.TensorLayout(("model", None), mesh)
    layout_map[".*position_embedding/embeddings"] = keras.distribution.TensorLayout((None, "model"), mesh)
    
    # MHA: Q/K/V shard on heads, Attention Output sharded on input dim to match
    layout_map[".*self_attention/query/kernel"] = keras.distribution.TensorLayout((None, "model", None), mesh)
    layout_map[".*self_attention/key/kernel"] = keras.distribution.TensorLayout((None, "model", None), mesh)
    layout_map[".*self_attention/value/kernel"] = keras.distribution.TensorLayout((None, "model", None), mesh)
    layout_map[".*self_attention/attention_output/kernel"] = keras.distribution.TensorLayout(("model", None), mesh)
    
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
    return layout_map

def run_training(rank, world_size, layout_map, backend):
    import keras
    import keras_hub
    
    gc.collect()
    tracker = MemoryTracker()
    base_cpu = psutil.Process(os.getpid()).memory_info().rss
    tracker.start()

    distribution = keras.distribution.ModelParallel(
        layout_map=layout_map, 
        batch_dim_name="data", 
        auto_shard_dataset=False
    )
    
    with distribution.scope():
        # Staggered loading to avoid system-wide memory spike in multi-process mode
        if backend == "torch":
            time.sleep(rank * 10) # Give previous ranks time to shard and GC
            
        model = keras_hub.models.OPTBackbone.from_preset("opt_125m_en", dropout=0.0)
        gc.collect()

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5), 
            loss="mse", 
            jit_compile=True
        )
        gc.collect()

        if backend == "torch":
            import torch
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
        
        np.random.seed(42)
        if backend == "torch":
            indices = []
            for i in range(10):
                base = i * 4
                indices.extend([base, base + 1] if rank < 2 else [base + 2, base + 3])
            
            # Optimization: only generate what we need or delete full arrays immediately
            full_token_ids = np.random.randint(0, 50272, (40, 32)).astype("int32")
            full_padding_mask = np.ones((40, 32), dtype="int32")
            full_y = np.random.normal(size=(40, 32, 768)).astype("float32")

            for i in range(10):
                base = i * 4
                full_token_ids[base+2:base+4] = full_token_ids[base:base+2]
                full_y[base+2:base+4] = full_y[base:base+2]
            
            x = {
                "token_ids": full_token_ids[indices],
                "padding_mask": full_padding_mask[indices]
            }
            y = full_y[indices]
            
            del full_token_ids, full_padding_mask, full_y
            gc.collect()
            
            # Explicitly convert to tensors on the correct device to avoid later replication
            import torch
            device_idx = int(os.environ.get("LOCAL_RANK", 0))
            device = torch.device(f"cuda:{device_idx}" if torch.cuda.is_available() else "cpu")
            x = {k: torch.from_numpy(v).to(device) for k, v in x.items()}
            y = torch.from_numpy(y).to(device)
            gc.collect()
            
            batch_size = 2
        else:
            x_full = {
                "token_ids": np.random.randint(0, 50272, (40, 32)).astype("int32"),
                "padding_mask": np.ones((40, 32), dtype="int32")
            }
            y_full = np.random.normal(size=(40, 32, 768)).astype("float32")

            for i in range(10):
                base = i * 4
                x_full["token_ids"][base+2:base+4] = x_full["token_ids"][base:base+2]
                y_full[base+2:base+4] = y_full[base:base+2]
            x, y = x_full, y_full
            batch_size = 4

        # Compilation warmup
        start_compilation = time.time()
        model.fit(x, y, batch_size=batch_size, epochs=1, steps_per_epoch=1, verbose=1 if rank == 0 else 0, shuffle=False)
        compilation_time = time.time() - start_compilation
        
        # We no longer reset here because we want to include sharded weights in the peak
        
        start_time = time.time()
        history = model.fit(x, y, batch_size=batch_size, epochs=5, steps_per_epoch=1, verbose=1 if rank == 0 else 0, shuffle=False)
        training_time = time.time() - start_time
        
        peak_absolute = tracker.stop()

        # Compare GPU memory specifically
        has_gpu = False
        if backend == "jax":
            import jax
            device_peaks = []
            for d in jax.local_devices():
                if d.platform == 'gpu':
                    has_gpu = True
                    # Get peak for each device separately
                    device_peaks.append(d.memory_stats()['peak_bytes_in_use'])
            
            # Use MAX to get the peak of the busiest device
            peak_mem_mb = max(device_peaks) / (1024 * 1024) if device_peaks else 0
        else:
            import torch
            if torch.cuda.is_available():
                has_gpu = True
                rank_peak_gpu = torch.cuda.max_memory_allocated()
                device_idx = int(os.environ.get("LOCAL_RANK", 0))
                device = torch.device(f"cuda:{device_idx}")
                m_tensor = torch.tensor([float(rank_peak_gpu)], device=device)
                # Use MAX to see the busiest GPU's peak
                torch.distributed.all_reduce(m_tensor, op=torch.distributed.ReduceOp.MAX)
                peak_mem_mb = m_tensor.item() / (1024 * 1024)
            else:
                peak_mem_mb = 0

        # Fallback to max of absolute CPU peaks if no GPU was found (for local testing)
        if not has_gpu:
            delta = float(peak_absolute - base_cpu)
            if backend == "torch":
                import torch
                p_tensor = torch.tensor([delta])
                # Use MAX to see the busiest process's peak delta (Weights + States + Activations)
                torch.distributed.all_reduce(p_tensor, op=torch.distributed.ReduceOp.MAX)
                peak_mem_mb = p_tensor.item() / (1024 * 1024)
            else:
                # Normalize JAX total delta by world_size to get per-device equivalent
                peak_mem_mb = (delta / world_size) / (1024 * 1024)

        if rank == 0:
            if os.path.exists(f"results_{backend}.json"):
                with open(f"results_{backend}.json", "r") as f:
                    try:
                        old_results = json.load(f)
                        old_peak = old_results.get("peak_memory_mb", 0.0)
                        # Keep max for both to see the true high-water mark during training
                        peak_mem_mb = max(old_peak, peak_mem_mb)
                    except json.JSONDecodeError:
                        pass
            step_1_loss = float(history.history["loss"][0])
            step_5_loss = float(history.history["loss"][4])
            results = {
                "step_1_loss": step_1_loss,
                "step_5_loss": step_5_loss,
                "perplexity": float(np.exp(step_5_loss)),
                "throughput": (4 * 5) / training_time,
                "compilation_time": compilation_time,
                "training_time": training_time,
                "peak_memory_mb": peak_mem_mb,
            }
            with open(f"results_{backend}.json", "w") as f: 
                json.dump(results, f, indent=2)

def run_backend(backend, world_size=4):
    os.environ["KERAS_BACKEND"] = backend
    if backend == "jax":
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
        num_gpus = 0
        try:
            import torch
            num_gpus = torch.cuda.device_count()
        except:
            pass
        
        if num_gpus < world_size:
            os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={world_size}"
            os.environ["JAX_PLATFORMS"] = "cpu"
        _run_jax(world_size)
    elif backend == "torch":
        import torch
        port = str(find_free_port())
        torch.multiprocessing.spawn(_run_torch, args=(world_size, port), nprocs=world_size, join=True)

def _run_jax(world_size):
    import keras
    keras.utils.set_random_seed(42)
    
    devices = keras.distribution.list_devices()
    if len(devices) > world_size:
        devices = devices[:world_size]
    print(f"Using JAX devices: {devices}")
    
    if len(devices) < world_size:
        raise ValueError(f"Not enough devices found. Expected {world_size}, got {len(devices)}.")

    mesh = keras.distribution.DeviceMesh(shape=(2, 2), axis_names=("data", "model"), devices=devices)
    run_training(0, world_size, get_layout_map(mesh), "jax")

def _run_torch(rank, world_size, port):
    import torch
    
    # Minimize per-process thread overhead
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


    os.environ.update({
        "RANK": str(rank),
        "WORLD_SIZE": str(world_size),
        "LOCAL_RANK": str(rank),
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": port,
    })
    
    num_gpus = torch.cuda.device_count()
    device_type = "cuda" if num_gpus >= world_size else "cpu"
    os.environ["KERAS_TORCH_DEVICE"] = device_type
    
    import keras
    keras.utils.set_random_seed(42)
    keras.distribution.initialize()
    
    print(f"[Rank {rank}] Initialized. World size: {world_size}")
    
    devices = keras.distribution.list_devices(device_type)[:world_size]
    mesh = keras.distribution.DeviceMesh(shape=(2, 2), axis_names=("data", "model"), devices=devices)
    run_training(rank, world_size, get_layout_map(mesh), "torch")
    
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    run_backend(sys.argv[1])
