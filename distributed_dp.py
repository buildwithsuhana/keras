import os
import sys
import numpy as np
import json
import time
import subprocess

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def run_backend(backend, world_size=2):
    os.environ["KERAS_BACKEND"] = backend
    if backend == "jax":
        os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={world_size}"
        _run_jax(world_size)
    elif backend == "torch":
        import torch
        port = str(np.random.randint(29500, 29999))
        torch.multiprocessing.spawn(_run_torch, args=(world_size, port), nprocs=world_size, join=True)

def _run_jax(world_size):
    import jax
    import keras
    import keras_hub
    keras.utils.set_random_seed(42)
    
    devices = keras.distribution.list_devices()[:world_size]
    mesh = keras.distribution.DeviceMesh(shape=(world_size,), axis_names=("batch",), devices=devices)
    
    distribution = keras.distribution.DataParallel(device_mesh=mesh, auto_shard_dataset=False)
    with distribution.scope():
        model = keras_hub.models.OPTBackbone.from_preset("opt_125m_en", dropout=0.0)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5), loss="mse")
        
        np.random.seed(42)
        num_samples = 8 
        x_full = {
            "token_ids": np.random.randint(0, 50272, (num_samples, 32)).astype("int32"),
            "padding_mask": np.ones((num_samples, 32), dtype="int32")
        }
        y_full = np.random.normal(size=(num_samples, 32, 768)).astype("float32")

        # Warmup
        model.fit(x_full, y_full, batch_size=4, epochs=1, steps_per_epoch=1, verbose=1, shuffle=False)
        
        start_time = time.time()
        history = model.fit({k: v[4:8] for k, v in x_full.items()}, y_full[4:8], 
                            batch_size=4, epochs=5, steps_per_epoch=1, verbose=1, shuffle=False)
        end_time = time.time()
        
        step_1_loss = float(history.history["loss"][0])
        step_5_loss = float(history.history["loss"][4])
        training_time = end_time - start_time
        throughput = (5 * 4) / training_time
        perplexity = float(np.exp(step_5_loss))

    results = {
        "step_1_loss": step_1_loss,
        "step_5_loss": step_5_loss,
        "perplexity": perplexity,
        "training_time": training_time,
        "throughput": throughput,
    }
    with open("results_jax_dp.json", "w") as f: json.dump(results, f, indent=2)

def _run_torch(rank, world_size, port):
    import os
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port
    os.environ["KERAS_TORCH_DEVICE"] = "cpu"
    import keras
    import keras_hub
    import torch
    import sys
    sys.path.insert(0, os.getcwd())
    keras.utils.set_random_seed(42)
    keras.distribution.initialize()
    
    devices = [f"cpu:{i}" for i in range(world_size)]
    mesh = keras.distribution.DeviceMesh(shape=(world_size,), axis_names=("batch",), devices=devices)
    
    distribution = keras.distribution.DataParallel(device_mesh=mesh, auto_shard_dataset=False)
    with distribution.scope():
        model = keras_hub.models.OPTBackbone.from_preset("opt_125m_en", dropout=0.0)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5), loss="mse")
        
        np.random.seed(42)
        num_samples = 8
        x_full = {
            "token_ids": np.random.randint(0, 50272, (num_samples, 32)).astype("int32"),
            "padding_mask": np.ones((num_samples, 32), dtype="int32")
        }
        y_full = np.random.normal(size=(num_samples, 32, 768)).astype("float32")
        
        indices = [rank * 2, rank * 2 + 1, 4 + rank * 2, 4 + rank * 2 + 1]
        x = {k: v[indices] for k, v in x_full.items()}
        y = y_full[indices]
        
        from torch.nn.attention import sdpa_kernel
        with sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            # Warmup
            model.fit({k: v[:2] for k, v in x.items()}, y[:2], 
                      batch_size=2, epochs=1, steps_per_epoch=1, verbose=1 if rank == 0 else 0, shuffle=False)
            
            start_time = time.time()
            history = model.fit({k: v[2:] for k, v in x.items()}, y[2:], 
                                batch_size=2, epochs=5, steps_per_epoch=1, verbose=1 if rank == 0 else 0, shuffle=False)
            end_time = time.time()
            
            step_1_loss = float(history.history["loss"][0])
            step_5_loss = float(history.history["loss"][4])
            training_time = end_time - start_time
            throughput = (5 * 4) / training_time
            perplexity = float(np.exp(step_5_loss))

    if rank == 0:
        results = {
            "step_1_loss": step_1_loss,
            "step_5_loss": step_5_loss,
            "perplexity": perplexity,
            "training_time": training_time,
            "throughput": throughput,
        }
        with open("results_torch_dp.json", "w") as f: json.dump(results, f, indent=2)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_backend(sys.argv[1])
    else:
        print("Running JAX backend...", flush=True)
        subprocess.run([sys.executable, __file__, "jax"], check=True)
        print("\nRunning Torch backend...", flush=True)
        subprocess.run([sys.executable, __file__, "torch"], check=True)
        
        try:
            with open("results_jax_dp.json", "r") as f: jax_res = json.load(f)
            with open("results_torch_dp.json", "r") as f: torch_res = json.load(f)
        except FileNotFoundError:
            print("Missing results files.")
            sys.exit(1)

        print("\n" + f"{'Metric':<30} | {'JAX':<20} | {'Torch':<20} | {'Diff':<15}")
        print("-" * 95)

        metrics = [
            ("Step 1 Loss", "step_1_loss"),
            ("Step 5 Loss", "step_5_loss"),
            ("Perplexity", "perplexity"),
            ("Throughput (samples/sec)", "throughput"),
            ("Training Time (sec)", "training_time"),
        ]

        all_pass = True
        for label, key in metrics:
            v_jax = jax_res[key]
            v_torch = torch_res[key]
            diff = abs(v_jax - v_torch)
            print(f"{label:<30} | {v_jax:<20.12f} | {v_torch:<20.12f} | {diff:<15.8e}")
            if key not in ["throughput", "training_time"] and diff > 1e-5:
                all_pass = False
