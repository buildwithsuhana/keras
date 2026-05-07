import json
import numpy as np

def compare():
    try:
        with open("results_jax.json", "r") as f: jax = json.load(f)
        with open("results_torch.json", "r") as f: torch = json.load(f)
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
    ]

    all_pass = True
    for label, key in metrics:
        v_jax = jax[key]
        v_torch = torch[key]
        diff = abs(v_jax - v_torch)
        print(f"{label:<30} | {v_jax:<20.12f} | {v_torch:<20.12f} | {diff:<15.8e}")
        if key not in ["throughput", "training_time"] and diff > 1e-5:
            all_pass = False

    print("\nSummary:")
    if all_pass:
        print("PASS: JAX and Torch results are numerically consistent.")
    else:
        print("FAIL: JAX and Torch results diverged beyond tolerance.")

if __name__ == "__main__":
    compare()