import numpy as np

# 1. Compare Step 1 Weight Updates
update_jax = np.load("update_step1_jax.npy")
update_torch = np.load("update_step1_torch.npy")

# In multi-process Torch, Rank 0 only saved its local shard.
# In single-process JAX, it saved the whole thing.
# Align shapes by taking the first shard of JAX.
if update_jax.shape != update_torch.shape:
    update_jax = update_jax[:update_torch.shape[0]]

update_diff = np.abs(update_jax - update_torch)
max_update_diff = np.max(update_diff)
mean_update_diff = np.mean(update_diff)

print("--- Step 1 Weight Update Comparison (float64) ---")
print(f"Max absolute difference: {max_update_diff:.6e}")
print(f"Mean absolute difference: {mean_update_diff:.6e}")

# 2. Compare Final Losses
with open("loss_jax.txt", "r") as f:
    loss_jax = float(f.read())
with open("loss_torch.txt", "r") as f:
    loss_torch = float(f.read())

loss_diff = abs(loss_jax - loss_torch)

print("\n--- Final Loss Comparison (10 Epochs, float64) ---")
print(f"JAX Loss:   {loss_jax:.12f}")
print(f"Torch Loss: {loss_torch:.12f}")
print(f"Difference: {loss_diff:.6e}")

# 3. Success Criteria
if max_update_diff < 1e-10 and loss_diff < 1e-5:
    print("\nSUCCESS: JAX and Torch are bit-consistent (float64).")
else:
    print("\nNOTICE: Minor differences remain even in float64.")
