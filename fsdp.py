import gc
import json
import os
import sys

# Ensure local keras is used
sys.path.insert(0, os.getcwd())

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
    import keras_hub

    import keras

    keras.backend.set_floatx("float32")

    gc.collect()
    tracker = MemoryTracker()
    base_cpu = psutil.Process(os.getpid()).memory_info().rss
    tracker.start()

    # Use FullyShardedDataParallel
    devices = keras.distribution.list_devices()
    if len(devices) > world_size:
        devices = devices[:world_size]

    mesh = keras.distribution.DeviceMesh(
        shape=(world_size,), axis_names=("batch",), devices=devices
    )

    distribution = keras.distribution.FullyShardedDataParallel(
        device_mesh=mesh,
        auto_shard_dataset=False,
        # We can also test min_num_params to trigger sharding for smaller layers
        min_num_params=1000,
    )

    with distribution.scope():
        if backend == "torch":
            time.sleep(rank * 1)

        model = keras_hub.models.OPTBackbone.from_preset(
            "opt_125m_en", dropout=0.0
        )
        gc.collect()

        # Check initial weights
        weights = model.get_weights()
        if rank == 0:
            print(
                f"[{backend}] Initial weight mean: {np.mean(weights[0]):.12f}"
            )

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5, epsilon=1e-7),
            loss="mse",
            jit_compile=False,
        )
        gc.collect()

        if backend == "torch":
            import torch

            if torch.distributed.is_initialized():
                torch.distributed.barrier()

        np.random.seed(42)
        base_batch_size = 16
        global_batch_size = base_batch_size * world_size
        num_total_samples = global_batch_size * 10

        full_token_ids = np.random.randint(
            0, 50272, (num_total_samples, 32)
        ).astype("int32")
        full_padding_mask = np.ones((num_total_samples, 32), dtype="int32")
        full_y = np.random.normal(size=(num_total_samples, 32, 768)).astype(
            "float32"
        )

        if backend == "torch":
            start_idx = rank * base_batch_size
            indices = []
            for i in range(10):
                base = i * global_batch_size + start_idx
                indices.extend(range(base, base + base_batch_size))

            x = {
                "token_ids": full_token_ids[indices],
                "padding_mask": full_padding_mask[indices],
            }
            y = full_y[indices]

            import torch

            device = torch.device("cpu")
            x = {k: torch.from_numpy(v).to(device) for k, v in x.items()}
            y = torch.from_numpy(y).to(device)
            batch_size = base_batch_size
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

        if rank == 0:
            weights_after = model.get_weights()
            print(
                f"[{backend}] Weight mean after warmup: {np.mean(weights_after[0]):.12f}"
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
                device = torch.device(
                    f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"
                )
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
                # In Torch, each process tracks its own RSS.
                # Dividing by world_size is NOT correct here because 'delta'
                # is already the memory of a single process (one replica).
                # However, to be consistent with JAX (which runs in one process
                # and we divide total delta by world_size), we should keep it
                # as is or adjust JAX.
                # Actually, the user asked why Torch is so high (4.8GB).
                # 4.8GB / 4 processes = 1.2GB, which is close to JAX's 1.3GB.
                # So the difference is that Torch reported the full process memory
                # while JAX reported the divided memory.
                peak_mem_mb = (p_tensor.item()) / (1024 * 1024)
            else:
                peak_mem_mb = (delta / world_size) / (1024 * 1024)

        if rank == 0:
            if os.path.exists(f"results_{backend}_fsdp.json"):
                with open(f"results_{backend}_fsdp.json", "r") as f:
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

            with open(f"results_{backend}_fsdp.json", "w") as f:
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
            os.environ["XLA_FLAGS"] = (
                f"--xla_force_host_platform_device_count={world_size}"
            )
            os.environ["JAX_PLATFORMS"] = "cpu"
        _run_jax(world_size)
    elif backend == "torch":
        os.environ["KERAS_TORCH_DEVICE"] = "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

        import torch

        port = str(find_free_port())
        torch.multiprocessing.spawn(
            _run_torch, args=(world_size, port), nprocs=world_size, join=True
        )


def _run_jax(world_size):
    import keras

    keras.utils.set_random_seed(42)
    run_training(0, world_size, "jax")


def _run_torch(rank, world_size, port):
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    os.environ.update(
        {
            "RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "LOCAL_RANK": str(rank),
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": port,
        }
    )

    if hasattr(torch, "set_default_device"):
        torch.set_default_device("cpu")

    import keras

    keras.utils.set_random_seed(42)
    keras.distribution.initialize()

    run_training(rank, world_size, "torch")

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fsdp.py <backend>")
        sys.exit(1)
    run_backend(sys.argv[1])
