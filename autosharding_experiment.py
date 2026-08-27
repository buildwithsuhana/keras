import gc
import json
import os
import sys
import threading
import time

import numpy as np
import psutil

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

def run_training(rank, world_size, backend):
    import keras
    import keras_hub
    import tensorflow as tf
    from keras.src.distribution.distribution_lib import AutoTPDistribution, DeviceMesh, list_devices

    keras.backend.set_floatx("float32")
    gc.collect()
    tracker = MemoryTracker()
    base_cpu = psutil.Process(os.getpid()).memory_info().rss
    tracker.start()

    # Create Logical Mesh
    if backend == "jax":
        devices = list_devices("cpu")[:world_size]
    else:
        devices = [f"cpu:{i}" for i in range(world_size)]
    
    device_mesh = DeviceMesh(shape=(world_size,), axis_names=("model",), devices=devices)

    # 1. Plan Generation (Fresh Instance)
    template_model = keras_hub.models.OPTBackbone.from_preset("opt_125m_en", dropout=0.0)
    # Build to resolve paths
    _ = template_model({"token_ids": np.ones((1, 32), "int32"), "padding_mask": np.ones((1, 32), "int32")})
    
    distribution = AutoTPDistribution(template_model, device_mesh=device_mesh)
    
    # 2. Set Global Distribution
    keras.distribution.set_distribution(distribution)

    # 3. Create Real Training Model
    model = keras_hub.models.OPTBackbone.from_preset("opt_125m_en", dropout=0.0)
    
    is_jit = True if backend == "jax" else False
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss="mse",
        jit_compile=is_jit,
    )
    
    # 4. Deterministic Data Generation
    np.random.seed(42)
    global_batch_size = 8
    steps = 5
    num_total_samples = global_batch_size * steps

    token_ids = np.random.randint(0, 50272, (num_total_samples, 32)).astype("int32")
    padding_mask = np.ones((num_total_samples, 32), dtype="int32")
    y_true = np.random.normal(size=(num_total_samples, 32, 768)).astype("float32")

    # Wrap in Dataset to satisfy Keras worker checks
    dataset = tf.data.Dataset.from_tensor_slices(
        ({"token_ids": token_ids, "padding_mask": padding_mask}, y_true)
    )

    if backend == "torch":
        # Manual sharding of the dataset for Torch workers
        dataset = dataset.shard(num_shards=world_size, index=rank)
        batch_size = global_batch_size // world_size
    else:
        # JAX uses global dataset with AutoTP
        batch_size = global_batch_size

    dataset = dataset.batch(batch_size)

    # Fit
    history = model.fit(
        dataset,
        epochs=1,
        steps_per_epoch=steps,
        verbose=1 if rank == 0 else 0,
    )
    
    training_time = time.time() - start_time if 'start_time' in locals() else 0
    # Actually fit doesn't return time, we measure it
    start_time = time.time()
    model.fit(dataset, epochs=1, steps_per_epoch=1, verbose=0)
    training_time = time.time() - start_time
    
    peak_absolute = tracker.stop()

    if rank == 0:
        delta = float(peak_absolute - base_cpu)
        peak_mem_mb = delta / (1024 * 1024)
        
        final_loss = float(history.history["loss"][-1])

        results = {
            "backend": backend,
            "final_loss": final_loss,
            "training_time": training_time,
            "peak_memory_mb": peak_mem_mb,
        }
        
        output_file = f"results_tp_{backend}.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results for TP {backend} saved to {output_file}")

def run_backend(backend, world_size=2):
    os.environ["KERAS_BACKEND"] = backend
    if backend == "jax":
        os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={world_size}"
        os.environ["JAX_PLATFORMS"] = "cpu"
        run_training(0, world_size, "jax")
    elif backend == "torch":
        os.environ["KERAS_TORCH_DEVICE"] = "cpu"
        import torch
        port = str(find_free_port())
        torch.multiprocessing.spawn(_run_torch, args=(world_size, port), nprocs=world_size, join=True)

def _run_torch(rank, world_size, port):
    os.environ.update({
        "RANK": str(rank), "WORLD_SIZE": str(world_size), "LOCAL_RANK": str(rank),
        "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": port,
        "KERAS_BACKEND": "torch", "KERAS_TORCH_DEVICE": "cpu"
    })
    import torch
    if hasattr(torch, "set_default_device"): torch.set_default_device("cpu")
    import keras
    keras.utils.set_random_seed(42)
    keras.distribution.initialize()
    run_training(rank, world_size, "torch")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2: sys.exit(1)
    run_backend(sys.argv[1])
