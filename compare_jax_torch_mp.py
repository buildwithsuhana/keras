import numpy as np
import os
import subprocess

def run_command(cmd, env=None):
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error:\n{result.stderr}")
    return result.stdout

def main():
    # 1. Generate fixed data
    subprocess.run(["python3", "generate_fixed_data.py"])
    
    # 2. Run JAX backend first (to save initial weights for sync)
    env_jax = os.environ.copy()
    env_jax["KERAS_BACKEND"] = "jax"
    env_jax["KERAS_DEVICE"] = "cpu"
    # JAX single process multi-device (it will use both GPUs/CPUs)
    env_jax["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    run_command(["python3", "compare_backend_worker.py"], env=env_jax)

    # 3. Run Torch backend
    env_torch = os.environ.copy()
    env_torch["KERAS_BACKEND"] = "torch"
    env_torch["KERAS_DEVICE"] = "cpu"
    env_torch["KERAS_TORCH_DEVICE"] = "cpu"
    # Use torchrun for 2 ranks
    run_command(["torchrun", "--nproc_per_node=2", "compare_backend_worker.py"], env=env_torch)
    
    # 4. Compare results
    print("\n" + "="*40)
    print("COMPARING NUMERICAL PARITY")
    print("="*40)
    
    try:
        for step in [0, 1, 2]:
            print(f"\n--- STEP {step} ---")
            for r in [0, 1]:
                t_file = f"torch_weights_step{step}_rank{r}.npy"
                j_file = f"jax_weights_step{step}_rank{r}.npy"

                if os.path.exists(t_file) and os.path.exists(j_file):
                    tw = np.load(t_file)
                    jw = np.load(j_file)

                    diff = np.abs(tw - jw)
                    print(f"\nComparison for Rank/Device {r}:")
                    print(f"  - Max absolute difference:  {np.max(diff):.8e}")
                    print(f"  - Mean absolute difference: {np.mean(diff):.8e}")
                    print(f"  - Torch mean: {np.mean(tw):.8f}")
                    print(f"  - JAX mean:   {np.mean(jw):.8f}")
                else:
                    print(f"\nMissing files for comparison of Step {step} Rank {r}")

    except Exception as e:
        print(f"Comparison failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
