import os
import sys
import subprocess
import numpy as np
import json

def run_backend(backend, script):
    print(f"\n--- Running {backend.upper()} backend ({script}) ---")
    cmd = [sys.executable, script, backend]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd() + (":" + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else "")
    result = subprocess.run(cmd, env=env)
    
    loss_file = f"loss_{backend}.json"
    if os.path.exists(loss_file):
        with open(loss_file, "r") as f:
            return json.load(f)["loss"]
    return None

if __name__ == "__main__":
    script = "run_dp_backend.py"
    for f in ["loss_torch.json", "loss_jax.json"]:
        if os.path.exists(f): os.remove(f)
            
    torch_loss = run_backend("torch", script)
    jax_loss = run_backend("jax", script)
    
    print("\n" + "="*30)
    print("DP RESULTS")
    print("="*30)
    
    if torch_loss is not None and jax_loss is not None:
        diff = abs(torch_loss - jax_loss)
        print(f"Torch loss: {torch_loss:.8f}")
        print(f"JAX loss:   {jax_loss:.8f}")
        print(f"Difference: {diff:.8e}")
        if diff <= 1e-5: print("\nSUCCESS: Divergence within tolerance!")
        else: print("\nFAILURE: Divergence exceeds tolerance!"); sys.exit(1)
    else: sys.exit(1)
