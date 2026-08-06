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
    return "29507"


def get_layout_map(mesh):
    import keras

    layout_map = keras.distribution.LayoutMap(mesh)

    # ==========================================
    # 🔠 Embedding Layers (Sharded)
    # ==========================================
    layout_map[".*token_embedding/embeddings"] = (
        keras.distribution.TensorLayout((None, "model"), mesh)
    )
    layout_map[".*position_embedding/embeddings"] = (
        keras.distribution.TensorLayout((None, "model"), mesh)
    )

    # ==========================================
    # 🧠 Attention Block (Sharded / Model Parallel)
    # ==========================================
    # QKV Projections: Shard along the Heads dimension (Axis 1)
    layout_map[".*self_attention/query/kernel"] = (
        keras.distribution.TensorLayout((None, "model", None), mesh)
    )
    layout_map[".*self_attention/key/kernel"] = keras.distribution.TensorLayout(
        (None, "model", None), mesh
    )
    layout_map[".*self_attention/value/kernel"] = (
        keras.distribution.TensorLayout((None, "model", None), mesh)
    )

    # Attention Dense Output: Shard along the Input Channel (Row Parallel)
    layout_map[".*self_attention/attention_output/kernel"] = (
        keras.distribution.TensorLayout(("model", None), mesh)
    )

    # ==========================================
    # 🚀 MLP (Feedforward) Block (Sharded)
    # ==========================================
    layout_map[".*feedforward_intermediate_dense/kernel"] = (
        keras.distribution.TensorLayout((None, "model"), mesh)
    )
    layout_map[".*feedforward_output_dense/kernel"] = (
        keras.distribution.TensorLayout(("model", None), mesh)
    )

    # ==========================================
    # ⚖️ Biases
    # ==========================================
    layout_map[".*self_attention/query/bias"] = keras.distribution.TensorLayout(
        (None,), mesh
    )
    layout_map[".*self_attention/key/bias"] = keras.distribution.TensorLayout(
        (None,), mesh
    )
    layout_map[".*self_attention/value/bias"] = keras.distribution.TensorLayout(
        (None,), mesh
    )
    layout_map[".*self_attention/attention_output/bias"] = (
        keras.distribution.TensorLayout((None,), mesh)
    )

    layout_map[".*feedforward_intermediate_dense/bias"] = (
        keras.distribution.TensorLayout(("model",), mesh)
    )
    layout_map[".*feedforward_output_dense/bias"] = (
        keras.distribution.TensorLayout((None,), mesh)
    )

    # ==========================================
    # 🛡️ Layer Normalization (Fully Replicated)
    # ==========================================
    layout_map[".*layer_norm/gamma"] = keras.distribution.TensorLayout(
        (None,), mesh
    )
    layout_map[".*layer_norm/beta"] = keras.distribution.TensorLayout(
        (None,), mesh
    )

    return layout_map


def run_training(rank, world_size, layout_map, backend):
    import keras_hub

    import keras

    print(f"[Rank {rank}] Entering run_training")
    keras.backend.set_floatx("float32")

    gc.collect()
    tracker = MemoryTracker()
    base_cpu = psutil.Process(os.getpid()).memory_info().rss
    tracker.start()

    print(f"[Rank {rank}] Creating ModelParallel distribution")
    distribution = keras.distribution.ModelParallel(
        layout_map=layout_map, batch_dim_name="data", auto_shard_dataset=False
    )

    with distribution.scope():
        print(f"[Rank {rank}] Entered distribution.scope()")
        if backend == "torch":
            time.sleep(rank * 1)

        print(f"[Rank {rank}] Creating model from preset")
        model = keras_hub.models.OPTBackbone.from_preset(
            "opt_125m_en", dropout=0.0
        )
        print(f"[Rank {rank}] Model created")
        gc.collect()

        is_jit = True if backend == "jax" else False
        print(f"[Rank {rank}] Compiling model")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5),
            loss="mse",
            jit_compile=is_jit,
        )
        print(f"[Rank {rank}] Model compiled")
        gc.collect()

        if backend == "torch":
            import torch

            if torch.distributed.is_initialized():
                print(f"[Rank {rank}] Waiting at barrier 1")
                torch.distributed.barrier()
                print(f"[Rank {rank}] Passed barrier 1")

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
            indices = []
            # mesh is (world_size, 1), axis_names=("data", "model")
            # data_axis_size = world_size, model_axis_size = 1
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

            import torch

            device = torch.device("cpu")
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
            batch_size = 32

        del full_token_ids, full_padding_mask, full_y
        gc.collect()

        # Warmup
        print(f"[Rank {rank}] Starting warmup fit")
        warmup_history = model.fit(
            {k: v[:batch_size] for k, v in x.items()},
            y[:batch_size],
            batch_size=batch_size,
            epochs=1,
            steps_per_epoch=1,
            verbose=1 if rank == 0 else 0,
            shuffle=False,
        )
        print(f"[Rank {rank}] Finished warmup fit")

        if backend == "torch" and torch.distributed.is_initialized():
            torch.distributed.barrier()
        start_time = time.time()
        epochs = 1
        steps_per_epoch = 5

        x_train = {k: v[batch_size:] for k, v in x.items()}
        y_train = y[batch_size:]

        print(f"[Rank {rank}] Starting training fit")
        history = model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            verbose=1 if rank == 0 else 0,
            shuffle=False,
        )
        print(f"[Rank {rank}] Finished training fit")

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
                peak_mem_mb = p_tensor.item() / (1024 * 1024)
            else:
                peak_mem_mb = (delta / world_size) / (1024 * 1024)

        if rank == 0:
            if os.path.exists(f"results_{backend}.json"):
                with open(f"results_{backend}.json", "r") as f:
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

            with open(f"results_{backend}.json", "w") as f:
                json.dump(results, f, indent=2)


def run_backend(backend, world_size=2):
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
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

        import torch

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
        shape=(world_size, 1), axis_names=("data", "model"), devices=devices
    )
    run_training(0, world_size, get_layout_map(mesh), "jax")


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

    # to maintain full backend-agnostic compliance.
    devices = [f"cpu:{i}" for i in range(world_size)]
    print(f"Using Torch devices: {devices}")

    mesh = keras.distribution.DeviceMesh(
        shape=(world_size, 1), axis_names=("data", "model"), devices=devices
    )
    run_training(rank, world_size, get_layout_map(mesh), "torch")

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python model_parallel_experiment.py <backend>")
        sys.exit(1)
    run_backend(sys.argv[1])
