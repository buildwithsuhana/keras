import os
import sys
import subprocess

# Set JAX as backend
os.environ["KERAS_BACKEND"] = "jax"

def get_jax_gpu_count():
    try:
        cmd = "import jax; print(len(jax.devices('gpu')))"
        res = subprocess.check_output([sys.executable, "-c", cmd], stderr=subprocess.DEVNULL)
        return int(res.decode().strip())
    except:
        return 0

gpu_count = get_jax_gpu_count()
has_gpu = gpu_count >= 2

if not has_gpu:
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    print("No 2+ GPUs found, using simulated CPU devices.")
else:
    print(f"Found {gpu_count} GPUs, using them.")

import jax
import time
import numpy as np
import keras
import keras_hub

def get_model():
    model = keras_hub.models.OPTBackbone.from_preset("opt_125m_en")
    model.compile(optimizer="adam", loss="mse")
    return model

def get_data(num_samples=256, seq_len=32, vocab_size=50272, embed_dim=768):
    x = {
        "token_ids": np.random.randint(0, vocab_size, (num_samples, seq_len)).astype("int32"),
        "padding_mask": np.ones((num_samples, seq_len), dtype="int32")
    }
    y = np.random.normal(size=(num_samples, seq_len, embed_dim)).astype("float32")
    return x, y

def measure_throughput(model, x, y, epochs=2, batch_size=8, verbose=0):
    # Warmup
    model.fit(x, y, epochs=1, batch_size=batch_size, verbose=0)
    
    start_time = time.time()
    model.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    end_time = time.time()
    
    total_samples = len(y) * epochs
    duration = end_time - start_time
    throughput = total_samples / duration
    return throughput

def main():
    print(f"JAX devices: {jax.devices()}")
    num_samples = 256
    batch_size = 8
    x, y = get_data(num_samples=num_samples)
    
    # Base case: 1 device, batch_size=8
    print(f"\n--- Single Device (BS={batch_size}) ---")
    keras.distribution.set_distribution(None)
    model_single = get_model()
    t_single = measure_throughput(model_single, x, y, batch_size=batch_size)
    print(f"Throughput: {t_single:.2f} samples/s")

    # Data Parallel: 2 devices, Global BS=16
    print(f"\n--- Data Parallel (2 devices, Global BS={batch_size*2}, verbose=0) ---")
    device_type = "gpu" if has_gpu else "cpu"
    devices = keras.distribution.list_devices(device_type)
    dp = keras.distribution.DataParallel(devices=devices[:2])
    with dp.scope():
        model_dp0 = get_model()
        t_dp0 = measure_throughput(model_dp0, x, y, batch_size=batch_size*2, verbose=0)
    print(f"Throughput (verbose=0): {t_dp0:.2f} samples/s")
    print(f"Scaling factor: {t_dp0 / t_single:.2f}x")

    # Data Parallel + steps_per_execution
    spe = 8
    print(f"\n--- Data Parallel (2 devices, Global BS={batch_size*2}, steps_per_execution={spe}) ---")
    with dp.scope():
        model_spe = get_model()
        model_spe.compile(optimizer="adam", loss="mse", steps_per_execution=spe)
        t_spe = measure_throughput(model_spe, x, y, batch_size=batch_size*2, verbose=0)
    print(f"Throughput (spe={spe}): {t_spe:.2f} samples/s")
    print(f"Scaling factor: {t_spe / t_single:.2f}x")

if __name__ == "__main__":
    main()
