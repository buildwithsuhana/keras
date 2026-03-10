import os
import numpy as np
import h5py

def load_weights_h5(path):
    weights = {}
    def visit_fn(name, obj):
        if isinstance(obj, h5py.Dataset):
            weights[name] = np.array(obj)
    with h5py.File(path, "r") as f:
        f.visititems(visit_fn)
    return weights

def compare(path1, path2):
    w1 = load_weights_h5(path1)
    w2 = load_weights_h5(path2)
    
    all_diffs = []
    print(f"{'Variable Path':<60} | {'Max Diff':<10}")
    print("-" * 75)
    
    common_keys = set(w1.keys()) & set(w2.keys())
    for k in sorted(common_keys):
        diff = np.abs(w1[k] - w2[k])
        max_diff = np.max(diff)
        all_diffs.append(max_diff)
        print(f"{k[:60]:<60} | {max_diff:.2e}")
        
    if not all_diffs:
        print("No common weights found to compare.")
        return

    total_max_diff = np.max(all_diffs)
    print("-" * 75)
    print(f"Overall Max Difference: {total_max_diff:.2e}")
    if total_max_diff < 1e-5:
        print("PASS: Divergence is within 1e-5")
    else:
        print("FAIL: Divergence exceeded 1e-5")

if __name__ == "__main__":
    print("Comparing JAX DP vs Torch DP:")
    compare("weights_jax_dp.weights.h5", "weights_torch_dp.weights.h5")
