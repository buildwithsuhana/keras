import os
import sys
import numpy as np
import json
import time
import subprocess

def find_free_port():
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]

def run_backend(backend, jit_compile, num_threads, world_size=2):
    os.environ["KERAS_BACKEND"] = backend
    if backend == "jax":
        os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={world_size}"
        os.environ["JAX_PLATFORMS"] = "cpu"
        # JAX threading flags
        if num_threads:
            os.environ["XLA_FLAGS"] += f" --xla_cpu_multi_thread_eigen={'true' if num_threads > 1 else 'false'}"
            os.environ["OMP_NUM_THREADS"] = str(num_threads)
            os.environ["MKL_NUM_THREADS"] = str(num_threads)
            os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)
        _run_jax(world_size, jit_compile)
    elif backend == "torch":
        import torch
        port = str(find_free_port())
        torch.multiprocessing.spawn(_run_torch, args=(world_size, port, jit_compile, num_threads), nprocs=world_size, join=True)

def _run_jax(world_size, jit_compile):
    import keras
    import keras_hub
    keras.utils.set_random_seed(42)
    devices = keras.distribution.list_devices()[:world_size]
    mesh = keras.distribution.DeviceMesh(shape=(world_size,), axis_names=("batch",), devices=devices)
    distribution = keras.distribution.DataParallel(device_mesh=mesh, auto_shard_dataset=False)
    
    with distribution.scope():
        model = keras_hub.models.OPTBackbone.from_preset("opt_125m_en", dropout=0.0)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5), loss="mse", jit_compile=jit_compile)

        global_batch_size = 32 * world_size
        num_samples = global_batch_size * 2
        x = {
            "token_ids": np.random.randint(0, 50272, (num_samples, 32)).astype("int32"),
            "padding_mask": np.ones((num_samples, 32), dtype="int32")
        }
        y = np.random.normal(size=(num_samples, 32, 768)).astype("float32")

        # Warmup
        model.fit(x, y, batch_size=global_batch_size, epochs=1, steps_per_epoch=1, verbose=0)

        start_time = time.time()
        model.fit(x, y, batch_size=global_batch_size, epochs=1, steps_per_epoch=1, verbose=0)
        end_time = time.time()
        
        throughput = global_batch_size / (end_time - start_time)
        print(f"JAX (jit={jit_compile}) throughput: {throughput:.2f} samples/sec")

def _run_torch(rank, world_size, port, jit_compile, num_threads):
    import os
    import torch
    if num_threads:
        torch.set_num_threads(num_threads)
    
    os.environ.update({"RANK": str(rank), "WORLD_SIZE": str(world_size), "LOCAL_RANK": str(rank), "MASTER_ADDR": "localhost", "MASTER_PORT": port, "KERAS_TORCH_DEVICE": "cpu"})
    
    import keras
    import keras_hub
    keras.utils.set_random_seed(42)
    keras.distribution.initialize()
    devices = keras.distribution.list_devices("cpu")[:world_size]
    mesh = keras.distribution.DeviceMesh(shape=(world_size,), axis_names=("batch",), devices=devices)
    distribution = keras.distribution.DataParallel(device_mesh=mesh, auto_shard_dataset=False)
    
    with distribution.scope():
        model = keras_hub.models.OPTBackbone.from_preset("opt_125m_en", dropout=0.0)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5), loss="mse", jit_compile=jit_compile)

        base_batch_size = 32
        x = {
            "token_ids": np.random.randint(0, 50272, (base_batch_size, 32)).astype("int32"),
            "padding_mask": np.ones((base_batch_size, 32), dtype="int32")
        }
        y = np.random.normal(size=(base_batch_size, 32, 768)).astype("float32")

        from torch.nn.attention import sdpa_kernel
        with sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            model.fit(x, y, batch_size=base_batch_size, epochs=1, steps_per_epoch=1, verbose=0)
            if torch.distributed.is_initialized(): torch.distributed.barrier()
            start_time = time.time()
            model.fit(x, y, batch_size=base_batch_size, epochs=1, steps_per_epoch=1, verbose=0)
            if torch.distributed.is_initialized(): torch.distributed.barrier()
            end_time = time.time()
            
            if rank == 0:
                throughput = (base_batch_size * world_size) / (end_time - start_time)
                print(f"Torch (jit={jit_compile}, threads={num_threads}) throughput: {throughput:.2f} samples/sec")

if __name__ == "__main__":
    # Test 1: JAX vs Torch (Default jit for each, Torch many threads)
    print("Test 1: Default configurations")
    subprocess.run([sys.executable, "-c", "import repro_throughput; repro_throughput.run_backend('jax', True, None)"])
    subprocess.run([sys.executable, "-c", "import repro_throughput; repro_throughput.run_backend('torch', False, None)"])
    
    # Test 2: JAX vs Torch (Both jit=True, Torch many threads)
    print("\nTest 2: Both jit=True")
    subprocess.run([sys.executable, "-c", "import repro_throughput; repro_throughput.run_backend('jax', True, None)"])
    subprocess.run([sys.executable, "-c", "import repro_throughput; repro_throughput.run_backend('torch', True, None)"])

    # Test 3: JAX vs Torch (Equal threads=1)
    print("\nTest 3: Both 1 thread")
    subprocess.run([sys.executable, "-c", "import repro_throughput; repro_throughput.run_backend('jax', True, 1)"])
    subprocess.run([sys.executable, "-c", "import repro_throughput; repro_throughput.run_backend('torch', False, 1)"])
