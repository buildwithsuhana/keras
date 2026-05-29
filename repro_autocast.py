
import os
os.environ["KERAS_BACKEND"] = "torch"

import keras
from keras import ops
import torch

from keras.src.backend.common.variables import AutocastScope

def test_autocast():
    print(f"Backend: {keras.backend.backend()}")
    
    with AutocastScope("float16"):
        # Test arange
        a = ops.arange(0, 5, dtype="float32")
        print(f"ops.arange(0, 5, dtype='float32'): {a.dtype}")
        
        # Test ones
        b = ops.ones((5,), dtype="float32")
        print(f"ops.ones((5,), dtype='float32'): {b.dtype}")
        
        # Test zeros
        c = ops.zeros((5,), dtype="float32")
        print(f"ops.zeros((5,), dtype='float32'): {c.dtype}")

if __name__ == "__main__":
    test_autocast()
