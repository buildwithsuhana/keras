import os
import sys
import numpy as np
import json
import time
import psutil
import threading

class MemoryTracker:
    def __init__(self):
        self.peak_cpu = 0
        self.peak_gpu = 0
        self.running = False
        self.thread = None

    def _track(self):
        import os
        process = psutil.Process(os.getpid())
        
        # Identify if we have torch/jax for GPU tracking
        has_torch_cuda = False
        has_jax_gpu = False
        try:
            import torch
            has_torch_cuda = torch.cuda.is_available()
        except:
            pass
        try:
            import jax
            has_jax_gpu = any(d.platform == 'gpu' for d in jax.devices())
        except:
            pass

        while self.running:
            try:
                # Track CPU RSS
                mem = process.memory_info().rss
                if mem > self.peak_cpu:
                    self.peak_cpu = mem
                
                # Track GPU VRAM
                if has_torch_cuda:
                    gpu_mem = torch.cuda.max_memory_allocated()
                    if gpu_mem > self.peak_gpu:
                        self.peak_gpu = gpu_mem
                elif has_jax_gpu:
                    import jax
                    # JAX doesn't have a simple 'max_memory_allocated' for the process,
                    # but we can sum the bytes_in_use across local devices.
                    total_gpu_mem = 0
                    for d in jax.local_devices():
                        if d.platform == 'gpu':
                            stats = d.memory_stats()
                            total_gpu_mem += stats['peak_bytes_in_use']
                    if total_gpu_mem > self.peak_gpu:
                        self.peak_gpu = total_gpu_mem
            except:
                break
            time.sleep(0.1)

    def start(self):
        self.peak_cpu = 0
        self.peak_gpu = 0
        self.running = True
        self.thread = threading.Thread(target=self._track, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        # Return total peak (CPU RSS + GPU VRAM) in MB
        return (self.peak_cpu + self.peak_gpu) / (1024 * 1024)

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
    
    tracker = MemoryTracker()
    tracker.start()

    distribution = keras.distribution.ModelParallel(
        layout_map=layout_map, 
        batch_dim_name="model", 
        auto_shard_dataset=False
    )
    
    with distribution.scope():
        model = keras_hub.models.OPTBackbone.from_preset("opt_125m_en", dropout=0.0)
        
        if backend == "torch" and rank == 0:
            print(f"\n[Rank {rank}] Verifying weight sharding:")
            from torch.distributed.tensor import DTensor
            for v in model.trainable_variables:
                val = v.value
                sharding_info = "✅ Sharded" if isinstance(val, DTensor) else "❌ Not Sharded"
                print(f"    {v.path:<60} | {sharding_info}")

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5), 
            loss="mse", 
            jit_compile=False
        )
        
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

        if backend == "torch":
            indices = []
            for i in range(10):
                 base = i * 4
                 indices.extend([base, base + 1] if rank == 0 else [base + 2, base + 3])
            x, y = {k: v[indices] for k, v in x_full.items()}, y_full[indices]
            batch_size = 2
        else:
            x, y = x_full, y_full
            batch_size = 4

        # Warmup
        model.fit(x, y, batch_size=batch_size, epochs=1, steps_per_epoch=1, verbose=1 if rank == 0 else 0, shuffle=False)
        
        start_time = time.time()
        history = model.fit(x, y, batch_size=batch_size, epochs=5, steps_per_epoch=1, verbose=1 if rank == 0 else 0, shuffle=False)
        training_time = time.time() - start_time
        
        peak_mem = tracker.stop()

        if backend == "torch":
            import torch
            # Aggregate peak memory across all ranks for a fair "Total System" comparison
            device = os.environ.get("KERAS_TORCH_DEVICE", "cpu")
            m_tensor = torch.tensor([peak_mem], device=device)
            torch.distributed.all_reduce(m_tensor, op=torch.distributed.ReduceOp.SUM)
            peak_mem = float(m_tensor.item())

        if rank == 0:
            step_1_loss = float(history.history["loss"][0])
            step_5_loss = float(history.history["loss"][4])
            results = {
                "step_1_loss": step_1_loss,
                "step_5_loss": step_5_loss,
                "perplexity": float(np.exp(step_5_loss)),
                "throughput": (4 * 5) / training_time,
                "training_time": training_time,
                "peak_memory_mb": peak_mem,
            }
            with open(f"results_{backend}.json", "w") as f: 
                json.dump(results, f, indent=2)

def run_backend(backend, world_size=2):
    os.environ["KERAS_BACKEND"] = backend
    if backend == "jax":
        # Disable JAX VRAM pre-allocation for accurate measurement
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
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
        torch.multiprocessing.spawn(_run_torch, args=(world_size,), nprocs=world_size, join=True)

def _run_jax(world_size):
    import keras
    keras.utils.set_random_seed(42)
    
    devices = keras.distribution.list_devices()
    if len(devices) > world_size:
        devices = devices[:world_size]
    print(f"Using JAX devices: {devices}")
    
    if len(devices) < world_size:
        raise ValueError(f"Not enough devices found. Expected {world_size}, got {len(devices)}.")

    mesh = keras.distribution.DeviceMesh(shape=(world_size,), axis_names=("model",), devices=devices)
    run_training(0, world_size, get_layout_map(mesh), "jax")

def _run_torch(rank, world_size):
    import os
    import torch


    os.environ.update({"RANK": str(rank), "WORLD_SIZE": str(world_size), "LOCAL_RANK": str(rank), "MASTER_ADDR": "localhost", "MASTER_PORT": "29560"})
    
    num_gpus = torch.cuda.device_count()
    device_type = "cuda" if num_gpus >= world_size else "cpu"
    os.environ["KERAS_TORCH_DEVICE"] = device_type
    
    import keras
    keras.utils.set_random_seed(42)
    keras.distribution.initialize()
    
    print(f"[Rank {rank}] Initialized. World size: {world_size}")
    
    devices = keras.distribution.list_devices(device_type)[:world_size]
    mesh = keras.distribution.DeviceMesh(shape=(world_size,), axis_names=("model",), devices=devices)
    run_training(rank, world_size, get_layout_map(mesh), "torch")
    
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    run_backend(sys.argv[1])
