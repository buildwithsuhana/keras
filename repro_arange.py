import os
os.environ["KERAS_BACKEND"] = "torch"
import keras
import torch
from keras.src.backend.torch.numpy import arange
from keras.src.backend.torch.core import convert_to_tensor

def test_arange_promotion():
    print("Testing arange promotion...")
    # Keras default floatx is usually float32
    print(f"Current floatx: {keras.backend.floatx()}")
    
    # arange with float start
    x = arange(0.0, 5.0)
    print(f"arange(0.0, 5.0) dtype: {x.dtype}")
    
    # arange with int start
    y = arange(0, 5)
    print(f"arange(0, 5) dtype: {y.dtype}")

def test_convert_to_tensor_promotion():
    print("\nTesting convert_to_tensor promotion...")
    
    # User data (list of floats)
    x = convert_to_tensor([1.0, 2.0])
    print(f"convert_to_tensor([1.0, 2.0]) dtype: {x.dtype}")
    
    # Op output (torch tensor float64)
    t64 = torch.tensor([1.0, 2.0], dtype=torch.float64)
    x = convert_to_tensor(t64)
    print(f"convert_to_tensor(torch.float64 tensor) dtype: {x.dtype}")

if __name__ == "__main__":
    test_arange_promotion()
    test_convert_to_tensor_promotion()
