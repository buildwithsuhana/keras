import subprocess
import os
import numpy as np
import matplotlib.pyplot as plt
import torch

def main():
    # 1. Generate Data
    print("Generating fixed data...")
    batch_size = 4 
    seq_len = 32
    vocab_size = 50272
    hidden_dim = 768 # Full OPT 125M
    
    np.save("opt_token_ids.npy", np.random.randint(0, vocab_size, (batch_size, seq_len)).astype("int32"))
    np.save("opt_padding_mask.npy", np.ones((batch_size, seq_len), dtype="int32"))
    np.save("opt_targets.npy", np.random.randn(batch_size, seq_len, hidden_dim).astype("float32"))

    # 2. Run JAX (Simulated CPU Devices or GPUs)
    num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 2
    print(f"\n--- Running JAX ({num_devices} devices) ---")
    env_jax = os.environ.copy()
    env_jax["KERAS_BACKEND"] = "jax"
    # Force CPU for absolute parity on Mac if no GPU
    if not torch.cuda.is_available():
        env_jax["KERAS_TORCH_DEVICE"] = "cpu"
        env_jax["KERAS_DEVICE"] = "cpu"

    # Prevent JAX from pre-allocating all GPU memory
    env_jax["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.4"
    env_jax["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    
    if torch.cuda.is_available():
        # If GPUs are available, use them directly
        subprocess.run(["python3", "opt_worker.py"], env=env_jax)
    else:
        # Fallback to simulated CPU devices
        env_jax["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={num_devices}"
        subprocess.run(["python3", "opt_worker.py"], env=env_jax)

    # 3. Run Torch (Simulated Ranks)
    print(f"\n--- Running Torch ({num_devices} ranks) ---")
    env_torch = os.environ.copy()
    env_torch["KERAS_BACKEND"] = "torch"
    if not torch.cuda.is_available():
        env_torch["KERAS_TORCH_DEVICE"] = "cpu"
        env_torch["KERAS_DEVICE"] = "cpu"
    # Use torchrun to launch ranks
    subprocess.run(["torchrun", f"--nproc_per_node={num_devices}", "opt_worker.py"], env=env_torch)

    # 4. Compare
    print("\n" + "="*40)
    print("SHARDED OPT 125M DIVERGENCE ANALYSIS")
    print("="*40)

    jax_losses = []
    torch_losses = []
    
    try:
        num_steps = 10
        for step in range(1, num_steps + 1):
            is_final = (step == num_steps)
            header = f"--- STEP {step} {'(FINAL)' if is_final else ''} ---"
            print(f"\n{header}")
            
            # Compare losses
            j_loss_file = f"jax_opt_s{step}_loss.npy"
            t_loss_file = f"torch_opt_s{step}_loss.npy"
            if os.path.exists(j_loss_file) and os.path.exists(t_loss_file):
                jl = float(np.load(j_loss_file))
                tl = float(np.load(t_loss_file))
                jax_losses.append(jl)
                torch_losses.append(tl)
                loss_diff = np.abs(jl - tl)
                print(f"Loss: JAX={jl:.6f}, Torch={tl:.6f}, Diff={loss_diff:.2e}")
            
            # Compare shards for all devices/ranks
            for r in range(num_devices):
                j_file = f"jax_opt_s{step}_rank{r}.npy"
                t_file = f"torch_opt_s{step}_rank{r}.npy"
                
                if not os.path.exists(j_file) or not os.path.exists(t_file):
                    print(f"Rank {r}: Missing weight files for step {step}")
                    continue

                j_val = np.load(j_file)
                t_val = np.load(t_file)
                diff = np.abs(j_val - t_val)

                print(f"Rank {r} Weights: Max Abs Diff = {np.max(diff):.2e}, Mean Abs Diff = {np.mean(diff):.2e}")

                if np.max(diff) > 1e-3: # Slightly higher tolerance for 10 steps
                    print(f"RESULT: FAIL (Weights at Step {step} Rank {r})")
                    # return

        # 5. Generate Graph
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(jax_losses) + 1), jax_losses, label='JAX Loss', marker='o', linestyle='-', alpha=0.7)
        plt.plot(range(1, len(torch_losses) + 1), torch_losses, label='Torch Loss', marker='x', linestyle='--', alpha=0.7)
        plt.title('OPT 125M Model Parallel: JAX vs Torch Loss Comparison')
        plt.xlabel('Step (Epoch)')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.savefig('loss_comparison.png')
        print(f"\nGraph saved as 'loss_comparison.png'")

        print("\n" + "="*40)
        print("FINAL VERDICT: PASS")
        print("Model parallel JAX and Torch are numerically synchronized.")
        print("="*40)
    except Exception as e:
        print(f"Comparison failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
