
import os

os.environ["KERAS_BACKEND"] = "torch"

import keras
import torch
import numpy as np

print(f"Keras floatx: {keras.config.floatx()}")

# Case 1: Python float
x = keras.ops.arange(0.0, 5.0)
print(f"arange(0.0, 5.0) dtype: {x.dtype}")

# Case 2: float64 inputs
start = torch.tensor(0.0, dtype=torch.float64)
stop = torch.tensor(5.0, dtype=torch.float64)
x = keras.ops.arange(start, stop)
print(f"arange(float64_tensor, float64_tensor) dtype: {x.dtype}")

# Case 3: float64 numpy inputs
start_np = np.array(0.0, dtype="float64")
stop_np = np.array(5.0, dtype="float64")
x = keras.ops.arange(start_np, stop_np)
print(f"arange(float64_np, float64_np) dtype: {x.dtype}")
