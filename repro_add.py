import os

os.environ["KERAS_BACKEND"] = "torch"
os.environ["KERAS_TORCH_DEVICE"] = "cpu"

import keras
import torch
import numpy as np

print(f"Keras floatx: {keras.config.floatx()}")

# Case 2: float64 inputs
x1 = torch.tensor([1.0], dtype=torch.float64, device="cpu")
x2 = torch.tensor([2.0], dtype=torch.float64, device="cpu")
y = keras.ops.add(x1, x2)
print(f"add(float64_tensor, float64_tensor) dtype: {y.dtype}")

# Case 3: arange
start = torch.tensor(0.0, dtype=torch.float64, device="cpu")
stop = torch.tensor(5.0, dtype=torch.float64, device="cpu")
x = keras.ops.arange(start, stop)
print(f"arange(float64_tensor, float64_tensor) dtype: {x.dtype}")
