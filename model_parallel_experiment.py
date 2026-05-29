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
            time.sleep(0.1)

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
    return 29500 + np.random.randint(0, 1000)

def get_layout_map(mesh):
    import keras
    layout_map = keras.distribution.LayoutMap(mesh)
    
    layout_map[".*token_embedding/embeddings"] = keras.distribution.TensorLayout(("model", None), mesh)
    layout_map[".*position_embedding/embeddings"] = keras.distribution.TensorLayout((None, "model"), mesh)
    layout_map[".*self_attention/query/kernel"] = keras.distribution.TensorLayout((None, "model", None), mesh)
    layout_map[".*self_attention/key/kernel"] = keras.distribution.TensorLayout((None, "model", None), mesh)
    layout_map[".*self_attention/value/kernel"] = keras.distribution.TensorLayout((None, "model", None), mesh)
    layout_map[".*self_attention/attention_output/kernel"] = keras.distribution.TensorLayout(("model", None), mesh)
    layout_map[".*feedforward_intermediate_dense/kernel"] = keras.distribution.TensorLayout((None, "model"), mesh)
    layout_map[".*feedforward_output_dense/kernel"] = keras.distribution.TensorLayout(("model", None), mesh)
    
    layout_map[".*self_attention/query/bias"] = keras.distribution.TensorLayout(("model", None), mesh)
    layout_map[".*self_attention/key/bias"] = keras.distribution.TensorLayout(("model", None), mesh)
    layout_map[".*self_attention/value/bias"] = keras.distribution.TensorLayout(("model", None), mesh)
    layout_map[".*self_attention/attention_output/bias"] = keras.distribution.TensorLayout((None,), mesh)
    layout_map[".*feedforward_intermediate_dense/bias"] = keras.distribution.TensorLayout(("model",), mesh)
    layout_map[".*feedforward_output_dense/bias"] = keras.distribution.TensorLayout((None,), mesh)
    
    layout_map[".*layer_norm/gamma"] = keras.distribution.TensorLayout((None,), mesh)
    layout_map[".*layer_norm/beta"] = keras.distribution.TensorLayout((None,), mesh)
    return layout_map

def run_training(rank, world_size, layout_map, backend):
    import keras
    import keras_hub
    
    # Force float32 for maximum precision sync
    keras.backend.set_floatx("float32")
    
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
        if backend == "torch":
            time.sleep(rank * 1)
            
        # Load model
        model = keras_hub.models.OPTBackbone.from_preset("opt_125m_en", dropout=0.0)
        gc.collect()

        # Compile - Disable JIT for Torch to ensure exact operation matching with JAX
        # torch.compile can introduce numerical variations (1e-3 to 1e-4) due to fusions.
        jit_compile = True if backend == "jax" else False
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5, epsilon=1e-7), 
            loss="mse", 
            jit_compile=jit_compile
        )
        gc.collect()

        if backend == "torch":
            import torch
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
        
        # EXACT Data Generation Sync
        np.random.seed(42)
        global_batch_size = 32
        num_total_samples = global_batch_size * 10
        
        full_x_ids = np.random.randint(0, 50272, (num_total_samples, 32)).astype("int32")
        full_x_mask = np.ones((num_total_samples, 32), dtype="int32")
        full_y = np.random.normal(size=(num_total_samples, 32, 768)).astype("float32")
        
        if backend == "torch":
            batch_size = global_batch_size // world_size
            # Shard data so each rank gets a unique slice of the global batch
            all_indices = []
            for i in range(10):
                start = i * global_batch_size + rank * batch_size
                end = start + batch_size
                all_indices.extend(np.arange(start, end))
            
            x = {
                "token_ids": full_x_ids[all_indices],
                "padding_mask": full_x_mask[all_indices]
            }
            y = full_y[all_indices]
            
            import torch
            device_idx = int(os.environ.get("LOCAL_RANK", 0))
            device = torch.device(f"cuda:{device_idx}" if torch.cuda.is_available() else "cpu")
            x = {k: torch.from_numpy(v).to(device) for k, v in x.items()}
            y = torch.from_numpy(y).to(device)
            gc.collect()
        else:
            batch_size = global_batch_size
            x = {"token_ids": full_x_ids, "padding_mask": full_x_mask}
            y = full_y

        # Warmup (Step 0)
        warmup_history = model.fit(
            {k: v[:batch_size] for k, v in x.items()}, 
            y[:batch_size], 
            batch_size=batch_size, epochs=1, steps_per_epoch=1, 
            verbose=1 if rank == 0 else 0, shuffle=False
        )
        step_0_loss = float(warmup_history.history["loss"][0])
        
        if backend == "torch" and torch.distributed.is_initialized():
            torch.distributed.barrier()
        
        start_time = time.time()
        epochs = 5 # Steps 1 to 5
        
        x_train = {k: v[batch_size:] for k, v in x.items()}
        y_train = y[batch_size:]

        history = model.fit(
            x_train, y_train, 
            batch_size=batch_size, epochs=epochs, steps_per_epoch=1, 
            verbose=1 if rank == 0 else 0, shuffle=False
        )
        
        if backend == "torch" and torch.distributed.is_initialized():
            torch.distributed.barrier()
        training_time = time.time() - start_time
        
        peak_absolute = tracker.stop()

        # Average Memory per Device
        delta = float(peak_absolute - base_cpu)
        if backend == "torch":
            import torch
            device = torch.device(f"cpu")
            p_tensor = torch.tensor([delta])
            torch.distributed.all_reduce(p_tensor, op=torch.distributed.ReduceOp.SUM)
            peak_mem_mb = (p_tensor.item() / world_size) / (1024 * 1024)
        else:
            peak_mem_mb = (delta / world_size) / (1024 * 1024)

        if rank == 0:
            step_1_loss = float(history.history["loss"][0])
            final_loss = float(history.history["loss"][4])
            
            total_samples = global_batch_size * epochs
            throughput = total_samples / training_time

            results = {
                "step_0_loss": step_0_loss,
                "step_1_loss": step_1_loss,
                "final_loss": final_loss,
                "perplexity": float(np.exp(final_loss)),
                "throughput": throughput,
                "training_time": training_time,
                "peak_memory_mb": peak_mem_mb,
            }

            with open(f"results_{backend}.json", "w") as f: 
                json.dump(results, f, indent=2)

def run_backend(backend, world_size=4):
    os.environ["KERAS_BACKEND"] = backend
    if backend == "jax":
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        num_gpus = 0
        try:
            import torch
            num_gpus = torch.cuda.device_count()
        except: pass
        
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
    
    mesh = keras.distribution.DeviceMesh(shape=(2, 2), axis_names=("data", "model"), devices=devices)
    run_training(0, world_size, get_layout_map(mesh), "jax")

def _run_torch(rank, world_size, port):
    import torch
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
    
    devices = keras.distribution.list_devices(device_type)[:world_size]
    mesh = keras.distribution.DeviceMesh(shape=(2, 2), axis_names=("data", "model"), devices=devices)
    run_training(rank, world_size, get_layout_map(mesh), "torch")
    
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python model_parallel_experiment.py <backend>")
        sys.exit(1)
    run_backend(sys.argv[1])
