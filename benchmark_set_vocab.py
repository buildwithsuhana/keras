import os
import threading
import time

import psutil


def get_gpu_memory(backend):
    try:
        if backend == "tensorflow":
            import tensorflow as tf

            # Returns dict with 'current' and 'peak' in bytes
            info = tf.config.experimental.get_memory_info("GPU:0")
            return info["peak"] / (1024**2)
        elif backend == "torch":
            import torch

            if torch.cuda.is_available():
                return torch.cuda.max_memory_allocated() / (1024**2)
        elif backend == "jax":
            # JAX memory tracking is more complex, returning 0 as placeholder
            # if not easily accessible in this context
            return 0
    except:
        pass
    return 0


def run_benchmark(backend):
    print(f"\n--- Benchmarking {backend} ---")
    os.environ["KERAS_BACKEND"] = backend

    import tensorflow as tf

    from keras.src.layers.preprocessing.string_lookup import StringLookup

    vocab_size = 500000  # Increased for more visible memory impact
    vocab = [str(i) for i in range(vocab_size)]
    vocab_tensor = tf.constant(vocab)

    layer = StringLookup()

    # Track RAM usage
    process = psutil.Process(os.getpid())
    peak_ram = 0

    def monitor_ram():
        nonlocal peak_ram
        while getattr(threading.current_thread(), "do_run", True):
            mem = process.memory_info().rss / (1024**2)
            if mem > peak_ram:
                peak_ram = mem
            time.sleep(0.01)

    monitor_t = threading.Thread(target=monitor_ram)
    monitor_t.do_run = True
    monitor_t.start()

    # Warmup
    layer.set_vocabulary(["a", "b", "c"])

    start_time = time.time()
    num_runs = 5
    for i in range(num_runs):
        layer.set_vocabulary(vocab_tensor)
    end_time = time.time()

    monitor_t.do_run = False
    monitor_t.join()

    total_time = end_time - start_time
    avg_time = total_time / num_runs
    throughput = (vocab_size * num_runs) / total_time
    gpu_peak = get_gpu_memory(backend)

    print(f"Results for {backend}:")
    print(f"  Vocab Size:    {vocab_size}")
    print(f"  Avg Time:       {avg_time:.4f}s")
    print(f"  Throughput:     {throughput:,.0f} tokens/s")
    print(f"  Peak RAM:       {peak_ram:.2f} MB")
    if gpu_peak > 0:
        print(f"  Peak GPU Mem:   {gpu_peak:.2f} MB")

    # Correctness check
    current_vocab = layer.get_vocabulary()
    if len(current_vocab) != vocab_size + 1:
        raise ValueError(
            f"Expected vocab size {vocab_size + 1}, got {len(current_vocab)}"
        )
    print("  Correctness:    PASSED")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        run_benchmark(sys.argv[1])
    else:
        for backend in ["tensorflow", "jax", "torch"]:
            os.system(f"python3 {__file__} {backend}")
