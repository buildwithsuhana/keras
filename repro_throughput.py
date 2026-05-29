import os

# Set backend to JAX
os.environ["KERAS_BACKEND"] = "jax"
# Simulate 2 CPUs
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

import time
import numpy as np
import jax
import keras
import keras_hub
import tensorflow as tf

def get_model():
    # Load OPT 125M model from Keras Hub without preprocessor
    model = keras_hub.models.OPTCausalLM.from_preset("opt_125m_en", preprocessor=None)
    return model

def get_dataset(batch_size, seq_length, num_batches):
    # Create a dummy dataset
    x = np.random.randint(0, 30000, size=(num_batches * batch_size, seq_length), dtype="int32")
    # For CausalLM, labels are often the same as inputs (internally handled or passed as same)
    # Here we just provide them explicitly.
    dataset = tf.data.Dataset.from_tensor_slices((x, x))
    dataset = dataset.batch(batch_size)
    return dataset

def benchmark(use_distribution=False):
    batch_size = 4
    seq_length = 32
    num_batches = 10
    
    if use_distribution:
        # Create a DataParallel distribution
        devices = jax.local_devices()
        dist = keras.distribution.DataParallel(devices=devices)
        keras.distribution.set_distribution(dist)
        print(f"Using DataParallel distribution with devices: {devices}")
    else:
        keras.distribution.set_distribution(None)
        print("Using standard (no distribution)")

    model = get_model()
    # Use a simple optimizer and loss
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        jit_compile=True
    )
    
    dataset = get_dataset(batch_size, seq_length, num_batches)
    
    # Warmup
    print("Warmup (1 batch)...")
    model.fit(dataset.take(1), epochs=1, verbose=0)
    
    print(f"Benchmarking {num_batches} batches...")
    start_time = time.time()
    model.fit(dataset, epochs=1, verbose=1)
    end_time = time.time()
    
    total_time = end_time - start_time
    throughput = (batch_size * num_batches) / total_time
    print(f"Total time: {total_time:.2f}s")
    print(f"Throughput: {throughput:.2f} samples/s")
    return throughput

if __name__ == "__main__":
    print(f"JAX local devices: {jax.local_devices()}")
    
    # Run standard benchmark
    tp_standard = benchmark(use_distribution=False)
    
    # Clear Keras session/state if possible to avoid interference
    keras.backend.clear_session()
    
    # Run data parallel benchmark
    tp_parallel = benchmark(use_distribution=True)
    
    print("\nSummary Results:")
    print(f"Standard Throughput: {tp_standard:.2f} samples/s")
    print(f"Data Parallel Throughput: {tp_parallel:.2f} samples/s")
    if tp_standard > 0:
        print(f"Increase: {(tp_parallel/tp_standard - 1)*100:.2f}%")
