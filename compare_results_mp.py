import json
import numpy as np

def compare():
    try:
        with open("results_jax.json", "r") as f: jax = json.load(f)
        with open("results_torch.json", "r") as f: torch_res = json.load(f)
    except FileNotFoundError:
        print("Missing results files.")
        return

    print(f"{'Metric':<30} | {'JAX':<20} | {'Torch':<20} | {'Diff':<15}")
    print("-" * 95)

    metrics = [
        ("Step 1 Loss", "step_1_loss"),
        ("Step 5 Loss", "step_5_loss"),
        ("Perplexity", "perplexity"),
        ("Throughput (samples/sec)", "throughput"),
        ("Training Time (sec)", "training_time"),
        ("Compilation Time (sec)", "compilation_time"),
        ("Total Host Memory (MB)", "peak_memory"),
    ]

    all_pass = True
    for label, key in metrics:
        v_jax = jax.get(key, 0.0)
        v_torch = torch_res.get(key, 0.0)
        diff = abs(v_jax - v_torch)
        print(f"{label:<30} | {v_jax:<20.12f} | {v_torch:<20.12f} | {diff:<15.8e}")
        if key not in ["throughput", "training_time", "compilation_time", "peak_memory"] and diff > 1e-5:
            all_pass = False

    # Per-device memory
    print("\nPer-Device Memory Usage (VRAM if GPU, RSS if CPU):")
    print("-" * 95)
    jax_dev = jax.get("per_device_memory", {})
    torch_dev = torch_res.get("per_device_memory", {})
    
    all_devices = sorted(list(set(list(jax_dev.keys()) + list(torch_dev.keys()))))
    for dev in all_devices:
        v_jax = jax_dev.get(dev, 0.0)
        v_torch = torch_dev.get(dev, 0.0)
        print(f"{dev:<30} | {v_jax:<20.2f} MB | {v_torch:<20.2f} MB | N/A")

    print("\nSummary:")
    if all_pass:
        print("PASS: JAX and Torch results are numerically consistent.")
    else:
        print("FAIL: JAX and Torch results diverged beyond tolerance.")

if __name__ == "__main__":
    compare()