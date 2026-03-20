import numpy as np
import sys
import os

def compare(mode="mp"):
    prefix = "dp_" if mode == "dp" else ""
    title = "DataParallel" if mode == "dp" else "ModelParallel"

    update_jax_file = f"{prefix}update_step1_jax.npy"
    update_torch_file = f"{prefix}update_step1_torch.npy"
    loss_jax_file = f"{prefix}loss_jax.txt"
    loss_torch_file = f"{prefix}loss_torch.txt"
    init_loss_jax_file = f"{prefix}initial_loss_jax.txt"
    init_loss_torch_file = f"{prefix}initial_loss_torch.txt"

    if not (os.path.exists(update_jax_file) and os.path.exists(update_torch_file)):
        print(f"Skipping {title} comparison (files not found).")
        return

    # 0. Compare Initial Loss (Verification of Sync)
    if os.path.exists(init_loss_jax_file) and os.path.exists(init_loss_torch_file):
        with open(init_loss_jax_file, "r") as f:
            init_loss_jax = float(f.read())
        with open(init_loss_torch_file, "r") as f:
            init_loss_torch = float(f.read())
        init_loss_diff = abs(init_loss_jax - init_loss_torch)
        print(f"\n--- {title} Initial Sync Check (Step 0) ---")
        print(f"JAX Init Loss:   {init_loss_jax:.12f}")
        print(f"Torch Init Loss: {init_loss_torch:.12f}")
        print(f"Difference:      {init_loss_diff:.6e}")

    # 1. Compare Step 1 Weight Updates
    update_jax = np.load(update_jax_file)
    update_torch = np.load(update_torch_file)

    # Correct slicing: Take the corner of JAX global array that matches Torch shard
    if update_jax.shape != update_torch.shape:
        slices = tuple(slice(0, s) for s in update_torch.shape)
        update_jax = update_jax[slices]

    update_diff = np.abs(update_jax - update_torch)
    max_update_diff = np.max(update_diff)
    mean_update_diff = np.mean(update_diff)

    print(f"\n--- Step 1 {title} Weight Update Comparison (float32) ---")
    print(f"Max absolute difference: {max_update_diff:.6e}")
    print(f"Mean absolute difference: {mean_update_diff:.6e}")

    # 2. Compare Final Losses
    with open(loss_jax_file, "r") as f:
        loss_jax = float(f.read())
    with open(loss_torch_file, "r") as f:
        loss_torch = float(f.read())

    loss_diff = abs(loss_jax - loss_torch)

    print(f"\n--- Final {title} Loss Comparison (10 Epochs, float32) ---")
    print(f"JAX Loss:   {loss_jax:.12f}")
    print(f"Torch Loss: {loss_torch:.12f}")
    print(f"Difference: {loss_diff:.6e}")

    # 3. Success Criteria
    if max_update_diff < 1e-5 and loss_diff < 5e-3:
        print(f"\nSUCCESS: JAX and Torch are consistent in {title} (float32).")
    else:
        print(f"\nNOTICE: Significant differences remain in {title} (float32).")

if __name__ == "__main__":
    compare("mp")
    compare("dp")
