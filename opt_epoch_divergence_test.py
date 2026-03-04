import subprocess
import os
import numpy as np
import matplotlib.pyplot as plt
import torch

def main():
    # 1. Generate Data (Larger for epochs)
    print("Generating fixed data...")
    num_samples = 16 
    seq_len = 32
    vocab_size = 50272
    hidden_dim = 768 
    
    np.save("opt_token_ids_large.npy", np.random.randint(0, vocab_size, (num_samples, seq_len)).astype("int32"))
    np.save("opt_padding_mask_large.npy", np.ones((num_samples, seq_len), dtype="int32"))
    np.save("opt_targets_large.npy", np.random.randn(num_samples, seq_len, hidden_dim).astype("float32"))

    num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 2
    
    # 2. Run JAX
    print("\n--- Running JAX (10 epochs) ---")
    env_jax = os.environ.copy()
    env_jax["KERAS_BACKEND"] = "jax"
    if not torch.cuda.is_available():
        env_jax["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={num_devices}"
    
    subprocess.run(["python3", "opt_epoch_worker.py"], env=env_jax)

    # 3. Run Torch
    print("\n--- Running Torch (10 epochs) ---")
    env_torch = os.environ.copy()
    env_torch["KERAS_BACKEND"] = "torch"
    
    subprocess.run(["torchrun", f"--nproc_per_node={num_devices}", "opt_epoch_worker.py"], env=env_torch)

    # 4. Compare
    print("\n" + "="*40)
    print("OPT 125M EPOCH DIVERGENCE ANALYSIS")
    print("="*40)

    jax_losses = []
    torch_losses = []
    
    num_epochs = 10
    for epoch in range(1, num_epochs + 1):
        j_loss_file = f"jax_opt_epoch_{epoch}_loss.npy"
        t_loss_file = f"torch_opt_epoch_{epoch}_loss.npy"
        
        if os.path.exists(j_loss_file) and os.path.exists(t_loss_file):
            jl = float(np.load(j_loss_file))
            tl = float(np.load(t_loss_file))
            jax_losses.append(jl)
            torch_losses.append(tl)
            loss_diff = np.abs(jl - tl)
            print(f"Epoch {epoch}: JAX Loss = {jl:.6f}, Torch Loss = {tl:.6f}, Diff = {loss_diff:.2e}")
        else:
            print(f"Epoch {epoch}: Missing loss files")

    # 5. Generate Graph
    if len(jax_losses) > 0:
        plt.figure(figsize=(10, 6))
        epochs_range = range(1, len(jax_losses) + 1)
        plt.plot(epochs_range, jax_losses, label='JAX Loss', marker='o', linestyle='-', alpha=0.7)
        plt.plot(epochs_range, torch_losses, label='Torch Loss', marker='x', linestyle='--', alpha=0.7)
        plt.title('OPT 125M Epoch Training: JAX vs Torch Loss Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('Average MSE Loss')
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.savefig('epoch_loss_comparison.png')
        print(f"\nGraph saved as 'epoch_loss_comparison.png'")

    # 6. Final Result
    if len(jax_losses) == num_epochs and len(torch_losses) == num_epochs:
        final_diff = np.abs(jax_losses[-1] - torch_losses[-1])
        print(f"\nFinal Epoch Loss Difference: {final_diff:.2e}")
        
        if final_diff < 1e-4:
            print("\nFINAL VERDICT: PASS")
        else:
            print("\nFINAL VERDICT: FAIL (Divergence too high)")
    else:
        print("\nFINAL VERDICT: FAIL (Missing results)")

if __name__ == "__main__":
    main()
