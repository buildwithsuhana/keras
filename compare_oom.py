import gc
import os
import time

import numpy as np
import psutil


def track_memory(func, *args, **kwargs):
    gc.collect()
    process = psutil.Process(os.getpid())
    base_mem = process.memory_info().rss / (1024**2)

    start_time = time.time()
    func(*args, **kwargs)
    end_time = time.time()

    peak_mem = process.memory_info().rss / (1024**2)
    return peak_mem - base_mem, end_time - start_time


def old_way(vocab_tensor):
    # Simulate the old logic: Convert to numpy immediately
    vocab_np = vocab_tensor.numpy()
    # Simulate some validation
    unique_np = np.unique(vocab_np)
    return unique_np


def new_way(vocab_tensor):
    # Simulate the new logic: Stay in TF tensors
    import tensorflow as tf

    unique_tf, _ = tf.unique(vocab_tensor)
    return unique_tf


if __name__ == "__main__":
    import tensorflow as tf

    # 1 Million tokens
    print("Creating 1,000,000 tokens...")
    vocab_list = [str(i) for i in range(1000000)]
    vocab_tensor = tf.constant(vocab_list)

    print("\n--- Comparing Memory Usage (Host RAM) ---")

    mem_old, time_old = track_memory(old_way, vocab_tensor)
    print("Old Way (NumPy/CPU):")
    print(f"  Peak RAM Increase: {mem_old:.2f} MB")
    print(f"  Time Taken:        {time_old:.4f}s")

    mem_new, time_new = track_memory(new_way, vocab_tensor)
    print("\nNew Way (Tensor/Device):")
    print(f"  Peak RAM Increase: {mem_new:.2f} MB")
    print(f"  Time Taken:        {time_new:.4f}s")

    print("\nSummary:")
    print(f"The New Way is {time_old / time_new:.1f}x faster.")
    print(f"The Old Way used {mem_old - mem_new:.2f} MB more Host RAM.")
