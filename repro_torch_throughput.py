import os
import time
import numpy as np
import torch

# Set Torch as backend
os.environ["KERAS_BACKEND"] = "torch"

import keras
import keras_hub

def get_model():
    model = keras_hub.models.OPTBackbone.from_preset("opt_125m_en")
    model.compile(optimizer="adam", loss="mse")
    return model

def get_data(num_samples=128, seq_len=32, vocab_size=50272, embed_dim=768):
    x = {
        "token_ids": np.random.randint(0, vocab_size, (num_samples, seq_len)).astype("int32"),
        "padding_mask": np.ones((num_samples, seq_len), dtype="int32")
    }
    y = np.random.normal(size=(num_samples, seq_len, embed_dim)).astype("float32")
    return x, y

def measure_throughput(model, x, y, epochs=2, batch_size=8):
    # Warmup
    model.fit(x, y, epochs=1, batch_size=batch_size, verbose=0)
    
    start_time = time.time()
    model.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=0)
    end_time = time.time()
    
    total_samples = len(y) * epochs
    duration = end_time - start_time
    throughput = total_samples / duration
    return throughput

def main():
    # Check if multiple GPUs are available for Torch
    device_count = torch.cuda.device_count()
    print(f"Torch visible devices: {device_count}")
    
    num_samples = 128
    batch_size = 8
    x, y = get_data(num_samples=num_samples)
    
    print("\n--- Single Device (BS=%d) ---" % batch_size)
    keras.distribution.set_distribution(None)
    model_single = get_model()
    t_single = measure_throughput(model_single, x, y, batch_size=batch_size)
    print(f"Throughput: {t_single:.2f} samples/s")

    if device_count > 1:
        print("\n--- Data Parallel (%d devices, Global BS=%d) ---" % (device_count, batch_size*device_count))
        devices = [f"cuda:{i}" for i in range(device_count)]
        dp = keras.distribution.DataParallel(devices=devices)
        with dp.scope():
            model_dp = get_model()
            t_dp = measure_throughput(model_dp, x, y, batch_size=batch_size*device_count)
        print(f"Throughput: {t_dp:.2f} samples/s")
        print(f"Scaling factor: {t_dp / t_single:.2f}x")
    else:
        print("\nSkipping Data Parallel as only one or zero GPUs are available for Torch.")

if __name__ == "__main__":
    main()
