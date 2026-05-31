import os
os.environ["KERAS_BACKEND"] = "torch"

import keras
import numpy as np
import torch

def test_linspace_mixed():
    print("Testing linspace(scalar, tensor)...")
    try:
        start = 0.0
        stop = np.array([10.0, 20.0]).astype("float32")
        out = keras.ops.linspace(start, stop, num=5)
        print(f"Output shape: {out.shape}")
        print(f"Output:\n{out}")
    except Exception as e:
        print(f"Failed: {e}")

    print("\nTesting linspace(tensor, scalar)...")
    try:
        start = np.array([0.0, 5.0]).astype("float32")
        stop = 10.0
        out = keras.ops.linspace(start, stop, num=5)
        print(f"Output shape: {out.shape}")
        print(f"Output:\n{out}")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    test_linspace_mixed()
