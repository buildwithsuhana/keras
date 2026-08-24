import subprocess
import re
import os

def run_and_parse(backend):
    print(f"🚀 Running {backend.upper()} backend...")
    env = os.environ.copy()
    env["KERAS_BACKEND"] = backend
    if backend == "torch":
        env["KERAS_TORCH_DEVICE"] = "cpu"
    if backend == "jax":
        env["JAX_PLATFORMS"] = "cpu"
        
    cmd = ["python3", "autosharding.py", "--backend", backend]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    output = result.stdout + result.stderr
    
    metrics = {}
    patterns = {
        "INITIAL_LOSS": r"INITIAL_LOSS: ([\d\.]+)",
        "FIT_LOSS": r"FIT_LOSS: ([\d\.]+)",
        "FINAL_LOSS_AFTER_FIT": r"FINAL_LOSS_AFTER_FIT: ([\d\.]+)"
    }
    
    for name, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            metrics[name] = float(match.group(1))
    
    return metrics

def compare():
    # Clean up old weights to ensure fresh synchronization
    if os.path.exists("shared_weights_10000.weights.h5"):
        os.remove("shared_weights_10000.weights.h5")
    
    jax_metrics = run_and_parse("jax")
    torch_metrics = run_and_parse("torch")
    
    print("" + "="*60)
    print(f"{'Metric':<25} | {'JAX':<15} | {'Torch':<15} | {'Diff':<10}")
    print("-" * 60)
    
    all_keys = sorted(set(jax_metrics.keys()) | set(torch_metrics.keys()))
    
    for key in all_keys:
        v_jax = jax_metrics.get(key, float('nan'))
        v_torch = torch_metrics.get(key, float('nan'))
        diff = abs(v_jax - v_torch)
        
        print(f"{key:<25} | {v_jax:<15.12f} | {v_torch:<15.12f} | {diff:<10.2e}")
    
    print("="*60)

if __name__ == "__main__":
    compare()
