import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
os.environ["KERAS_BACKEND"] = "jax"

import time
import numpy as np
import jax
import keras
import keras_hub
from keras.distribution import DeviceMesh, DataParallel

# Verify devices
print(f"Available devices: {jax.local_devices()}")

def get_model():
    model = keras_hub.models.OPTBackbone.from_preset("opt_125m_en")
    return model

def measure_throughput_fit(model, distribution=None, num_steps=10, batch_size=4):
    # Dummy data
    seq_length = 128
    x = {
        "token_ids": np.ones((batch_size * num_steps, seq_length), dtype="int32"),
        "padding_mask": np.ones((batch_size * num_steps, seq_length), dtype="int32"),
    }
    y = np.ones((batch_size * num_steps, seq_length, 768), dtype="float32")

    # Simple loss and optimizer
    model.compile(optimizer="adam", loss="mse", jit_compile=True)

    # Warmup (1 epoch of 1 step)
    print("Warming up...")
    model.fit(
        {"token_ids": x["token_ids"][:batch_size], "padding_mask": x["padding_mask"][:batch_size]},
        y[:batch_size],
        epochs=1,
        verbose=0
    )
    
    print(f"Measuring throughput for {num_steps} steps using model.fit...")
    start_time = time.time()
    model.fit(x, y, epochs=1, batch_size=batch_size, verbose=0)
    end_time = time.time()
    
    total_time = end_time - start_time
    throughput = (num_steps * batch_size) / total_time
    return throughput

# 1. Measure on 1 device (default)
print("\n--- Running on 1 device ---")
model_1 = get_model()
throughput_1 = measure_throughput_fit(model_1, num_steps=10, batch_size=16)
print(f"Throughput (1 device): {throughput_1:.2f} samples/s")

# 2. Measure on 2 devices (Data Parallel)
print("\n--- Running on 2 devices (DataParallel) ---")
devices = jax.local_devices()
mesh = DeviceMesh(shape=(2,), axis_names=("batch",), devices=devices)
distribution = DataParallel(mesh)

with distribution.scope():
    model_2 = get_model()
    throughput_2 = measure_throughput_fit(model_2, distribution=distribution, num_steps=10, batch_size=16)
    print(f"Throughput (2 devices): {throughput_2:.2f} samples/s")

print(f"\nSpeedup: {throughput_2 / throughput_1:.2f}x")
