import os
import sys
import numpy as np

def run_jax():
    os.environ["KERAS_BACKEND"] = "jax"
    import keras
    import jax.numpy as jnp
    
    # Identical logits and targets
    logits = np.arange(10).reshape((1, 2, 5)).astype("float32")
    targets = np.array([[0, 1]], dtype="int32")
    
    loss = keras.ops.nn.sparse_categorical_crossentropy(targets, logits, from_logits=True)
    np.save("jax_loss.npy", np.array(loss))

def run_torch():
    os.environ["KERAS_BACKEND"] = "torch"
    os.environ["KERAS_TORCH_DEVICE"] = "cpu"
    import keras
    import torch
    
    logits = np.arange(10).reshape((1, 2, 5)).astype("float32")
    targets = np.array([[0, 1]], dtype="int32")
    
    loss = keras.ops.nn.sparse_categorical_crossentropy(targets, logits, from_logits=True)
    np.save("torch_loss.npy", loss.detach().cpu().numpy())

if __name__ == "__main__":
    import subprocess
    if len(sys.argv) > 1:
        if sys.argv[1] == "jax": run_jax()
        else: run_torch()
    else:
        subprocess.run([sys.executable, sys.argv[0], "jax"])
        subprocess.run([sys.executable, sys.argv[0], "torch"])
        jax_loss = np.load("jax_loss.npy")
        torch_loss = np.load("torch_loss.npy")
        print(f"JAX loss: {jax_loss}")
        print(f"Torch loss: {torch_loss}")
        print(f"Diff: {jax_loss - torch_loss}")
