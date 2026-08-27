import gc
import json
import os
import sys
import threading
import time

import numpy as np
import psutil

# Explicitly disable X64 for JAX to ensure pure float32
os.environ["JAX_ENABLE_X64"] = "False"

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
    import time
    from keras.src.distribution.distribution_lib import AutoTPDistribution, DeviceMesh, list_devices

    # 0. SETUP PRECISION TO FLOAT32
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

    # 1. Plan Generation
    template_model = keras_hub.models.OPTBackbone.from_preset("opt_125m_en", dropout=0.0)
    _ = template_model({"token_ids": np.ones((1, 32), "int32"), "padding_mask": np.ones((1, 32), "int32")})
    
    if rank == 0:
        print("🔍 Model Layers:")
        for layer in template_model._flatten_layers(recursive=True):
            print(f"   - Name: {layer.name}, Path: {layer.path}, Class: {layer.__class__.__name__}")
    
    distribution = AutoTPDistribution(template_model, device_mesh=device_mesh)
    
    # 2. Set Global Distribution
    keras.distribution.set_distribution(distribution)

    # 3. Create Real Training Model
    model = keras_hub.models.OPTBackbone.from_preset("opt_125m_en", dropout=0.0)
    
    # 4. Sync Weights
    weights_file = "tp_experiment_sync_f32.weights.h5"
    if backend == "jax":
        model.save_weights(weights_file)
    else:
        import time
        max_retries = 30
        while not os.path.exists(weights_file) and max_retries > 0:
            time.sleep(1)
            max_retries -= 1
        model.load_weights(weights_file)

    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=0.0),
        loss="mse",
        jit_compile=False,
    )
    
    # 5. Deterministic Data Generation (float32)
    np.random.seed(42)
    token_ids = np.random.randint(0, 50272, (8, 32)).astype("int32")
    padding_mask = np.ones((8, 32), dtype="int32")
    y_true = np.random.normal(size=(8, 32, 768)).astype("float32")

    x = {"token_ids": token_ids, "padding_mask": padding_mask}
    y = y_true

    # 6. Evaluation (Initial Loss)
    start_time = time.time()
    loss = model.test_on_batch(x, y)
    training_time = time.time() - start_time
    peak_absolute = tracker.stop()

    if rank == 0:
        delta = float(peak_absolute - base_cpu)
        peak_mem_mb = delta / (1024 * 1024)
        
        initial_loss = float(keras.ops.convert_to_numpy(loss))

        results = {
            "backend": backend,
            "initial_loss": initial_loss,
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
