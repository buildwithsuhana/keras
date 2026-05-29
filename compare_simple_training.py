import json
import numpy as np

def compare():
    try:
        with open("results_simple_jax.json", "r") as f: jax = json.load(f)
        with open("results_simple_torch.json", "r") as f: torch = json.load(f)
    except FileNotFoundError:
        print("Missing results files. Run simple_experiment.py for both backends first.")
        return

    print(f"{'Metric':<30} | {'JAX':<20} | {'Torch':<20} | {'Diff':<15}")
    print("-" * 95)

    metrics = [
        ("Final Loss", "final_loss"),
        ("Perplexity", "perplexity"),
        ("Throughput (samples/sec)", "throughput"),
        # ("Training Time (sec)", "training_time"),
    ]

    all_pass = True
    for label, key in metrics:
        v_jax = jax[key]
        v_torch = torch[key]
        diff = abs(v_jax - v_torch)
        print(f"{label:<30} | {v_jax:<20.12f} | {v_torch:<20.12f} | {diff:<15.8e}")
        if key not in ["throughput"] and diff > 1e-5:
            all_pass = False

    print("\nSummary:")
    if all_pass:
        print("PASS: JAX and Torch simple results are numerically consistent.")
    else:
        print("FAIL: JAX and Torch simple results diverged beyond tolerance.")

if __name__ == "__main__":
    compare()
