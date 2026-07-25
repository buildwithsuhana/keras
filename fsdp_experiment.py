import gc
import json
import os
import sys
import threading
import time

import numpy as np
import psutil

# Disable MPS only on macOS as a fallback, but allow CUDA
try:
    import torch

    if sys.platform == "darwin" and hasattr(torch.backends, "mps"):
        torch.backends.mps.is_available = lambda: False
except ImportError:
    pass


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


def run_training(rank, world_size, mesh, backend):
    import keras_hub

    import keras

    keras.backend.set_floatx("float32")

    gc.collect()
    tracker = MemoryTracker()
    base_cpu = psutil.Process(os.getpid()).memory_info().rss
    tracker.start()

    # Using FSDP distribution
    distribution = keras.distribution.FSDP(
        device_mesh=mesh, auto_shard_dataset=False
    )

    with distribution.scope():
        if backend == "torch":
            time.sleep(rank * 0.1)

        # Use a model large enough to see sharding benefits
        model = keras_hub.models.OPTBackbone.from_preset(
            "opt_125m_en", dropout=0.0
        )
        gc.collect()

        is_jit = True if backend == "jax" else False
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5),
            loss="mse",
            jit_compile=is_jit,
        )
        gc.collect()

        if backend == "torch":
            import torch

            if torch.distributed.is_initialized():
                torch.distributed.barrier()

        np.random.seed(42)
        global_batch_size = 32
        num_total_samples = global_batch_size * 10

        full_token_ids = np.random.randint(
            0, 50272, (num_total_samples, 32)
        ).astype("int32")
        full_padding_mask = np.ones((num_total_samples, 32), dtype="int32")
        full_y = np.random.normal(size=(num_total_samples, 32, 768)).astype(
            "float32"
        )

        if backend == "torch":
            import torch

            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

            indices = []
            # For FSDP, we shard across the whole world_size for data
            data_shard_index = rank
            local_batch_size = global_batch_size // world_size
            for i in range(10):
                base = i * global_batch_size
                start = base + data_shard_index * local_batch_size
                end = start + local_batch_size
                indices.extend(np.arange(start, end))

            x = {
                "token_ids": full_token_ids[indices],
                "padding_mask": full_padding_mask[indices],
            }
            y = full_y[indices]

            x = {k: torch.from_numpy(v).to(device) for k, v in x.items()}
            y = torch.from_numpy(y).to(device)
            batch_size = local_batch_size
        else:
            x, y = (
                {
                    "token_ids": full_token_ids,
                    "padding_mask": full_padding_mask,
                },
                full_y,
            )
            batch_size = global_batch_size

        del full_token_ids, full_padding_mask, full_y
        gc.collect()

        # Warmup
        warmup_history = model.fit(
            {k: v[:batch_size] for k, v in x.items()},
            y[:batch_size],
            batch_size=batch_size,
            epochs=1,
            steps_per_epoch=1,
            verbose=1 if rank == 0 else 0,
            shuffle=False,
        )

        if backend == "torch" and torch.distributed.is_initialized():
            torch.distributed.barrier()
        start_time = time.time()
        epochs = 1
        steps_per_epoch = 5

        x_train = {k: v[batch_size:] for k, v in x.items()}
        y_train = y[batch_size:]

        history = model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            verbose=1 if rank == 0 else 0,
            shuffle=False,
        )

        if backend == "torch" and torch.distributed.is_initialized():
            torch.distributed.barrier()
        training_time = time.time() - start_time

        peak_absolute = tracker.stop()

        has_gpu = False
        if backend == "jax":
            import jax

            device_peaks = [
                d.memory_stats()["peak_bytes_in_use"]
                for d in jax.local_devices()
                if d.platform == "gpu"
            ]
            has_gpu = len(device_peaks) > 0
            peak_mem_mb = max(device_peaks) / (1024 * 1024) if has_gpu else 0
        else:
            import torch

            if torch.cuda.is_available():
                has_gpu = True
                rank_peak_gpu = torch.cuda.max_memory_allocated()
                device_id = torch.cuda.current_device()
                device = torch.device(f"cuda:{device_id}")
                m_tensor = torch.tensor([float(rank_peak_gpu)], device=device)
                torch.distributed.all_reduce(
                    m_tensor, op=torch.distributed.ReduceOp.MAX
                )
                peak_mem_mb = m_tensor.item() / (1024 * 1024)
            else:
                peak_mem_mb = 0

        if not has_gpu:
            delta = float(peak_absolute - base_cpu)
            if backend == "torch":
                import torch

                p_tensor = torch.tensor([delta])
                torch.distributed.all_reduce(
                    p_tensor, op=torch.distributed.ReduceOp.MAX
                )
                peak_mem_mb = p_tensor.item() / (1024 * 1024)
            else:
                peak_mem_mb = (delta / world_size) / (1024 * 1024)

        if rank == 0:
            if os.path.exists(f"fsdp_results_{backend}.json"):
                with open(f"fsdp_results_{backend}.json", "r") as f:
                    try:
                        old_peak = json.load(f).get("peak_memory_mb", 0.0)
                        peak_mem_mb = max(old_peak, peak_mem_mb)
                    except json.JSONDecodeError:
                        pass

            step_0_loss = float(warmup_history.history["loss"][0])
            step_1_loss = float(history.history["loss"][0])
            final_loss = float(history.history["loss"][-1])

            total_samples = global_batch_size * steps_per_epoch * epochs
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

            with open(f"fsdp_results_{backend}.json", "w") as f:
                json.dump(results, f, indent=2)


def run_backend(backend, world_size=None):
    os.environ["KERAS_BACKEND"] = backend

    import torch

    num_gpus = torch.cuda.device_count()

    if world_size is None:
        if backend == "torch" and num_gpus > 0:
            world_size = num_gpus
        else:
            world_size = 4  # Default simulation

    if backend == "jax":
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
        if num_gpus < world_size:
            os.environ["XLA_FLAGS"] = (
                f"--xla_force_host_platform_device_count={world_size}"
            )
            os.environ["JAX_PLATFORMS"] = "cpu"
        _run_jax(world_size)
    elif backend == "torch":
        if num_gpus == 0:
            print("Warning: No GPU detected. Torch FSDP requires a GPU.")
            # We'll try to run on CPU anyway, but FSDP will likely fail
            os.environ["KERAS_TORCH_DEVICE"] = "cpu"

        port = str(find_free_port())
        torch.multiprocessing.spawn(
            _run_torch, args=(world_size, port), nprocs=world_size, join=True
        )


def _run_jax(world_size):
    import keras

    keras.utils.set_random_seed(42)
    devices = keras.distribution.list_devices()
    if len(devices) > world_size:
        devices = devices[:world_size]
    print(f"Using JAX devices: {devices}")

    mesh = keras.distribution.DeviceMesh(
        shape=(world_size,), axis_names=("data",), devices=devices
    )
    run_training(0, world_size, mesh, "jax")


def _run_torch(rank, world_size, port):
    import torch

    os.environ.update(
        {
            "RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "LOCAL_RANK": str(rank),
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": port,
        }
    )

    import keras

    keras.utils.set_random_seed(42)

    if torch.cuda.is_available():
        # Set device for this process
        device_id = rank % torch.cuda.device_count()
        torch.cuda.set_device(device_id)
        # Force Keras to use the correct GPU device
        # This is a bit of a hack to ensure the right device is picked up
        os.environ["KERAS_TORCH_DEVICE"] = f"cuda:{device_id}"
    else:
        os.environ["KERAS_TORCH_DEVICE"] = "cpu"

    keras.distribution.initialize()

    if torch.cuda.is_available():
        devices = [
            f"cuda:{i % torch.cuda.device_count()}" for i in range(world_size)
        ]
    else:
        devices = [f"cpu:{i}" for i in range(world_size)]

    print(f"Process {rank}: Using Torch devices: {devices}")

    mesh = keras.distribution.DeviceMesh(
        shape=(world_size,), axis_names=("data",), devices=devices
    )
    run_training(rank, world_size, mesh, "torch")

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    backend = sys.argv[1]
    world_size = int(sys.argv[2]) if len(sys.argv) > 2 else None
    run_backend(backend, world_size)
