import os
import sys
import subprocess
import numpy as np
import json

def run_backend(backend):
    print(f"\n--- Running {backend.upper()} backend ---")
    cmd = [sys.executable, "run_mp_backend.py", backend]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd() + (":" + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else "")
    
    result = subprocess.run(cmd, env=env)
    
    loss_file = f"loss_{backend}.json"
    if os.path.exists(loss_file):
        with open(loss_file, "r") as f:
            return json.load(f)["loss"]
    return None

if __name__ == "__main__":
    for f in ["loss_torch.json", "loss_jax.json"]:
        if os.path.exists(f): os.remove(f)
            
    torch_loss = run_backend("torch")
    jax_loss = run_backend("jax")
    
    print("\n" + "="*30)
    print("MP RESULTS")
    print("="*30)
    
    if torch_loss is not None and jax_loss is not None:
        # In this specific test:
        # JAX fit() returns mean(L0, L1)
        # Torch fit() was updated to return mean(L0, L1) from its processes.
        # So they should now be directly comparable.
        diff = abs(torch_loss - jax_loss)
        print(f"Torch loss: {torch_loss:.8f}")
        print(f"JAX loss:   {jax_loss:.8f}")
        print(f"Difference: {diff:.8e}")
        
        if diff <= 1e-5:
            print("\nSUCCESS: Divergence is within tolerance!")
        else:
            # Check if it is the sharded vs mean reporting difference
            # In our case L0=7.56816959, L1=7.5276...
            # Mean is ~7.5479.
            # If the difference is ~0.02, it means one reported L0 and other mean.
            print("\nFAILURE: Divergence exceeds tolerance!")
            sys.exit(1)
    else:
        sys.exit(1)
