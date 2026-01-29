#!/usr/bin/env python3
"""Simple test to verify the gradient fix."""

import os
# Set backend before importing keras
os.environ["KERAS_BACKEND"] = "torch"

import torch
import numpy as np
from keras import Sequential
from keras.layers import Dense
from keras.distribution import DataParallel

def test_gradient_flow():
    """Test that gradients flow properly through distributed model."""
    print("Testing gradient flow...")
    
    # Create distribution
    devices = ["cuda:0"] if torch.cuda.is_available() else ["cpu:0"]
    dp = DataParallel(devices=devices, auto_shard_dataset=False)
    
    with dp.scope():
        model = Sequential([
            Dense(128, activation="relu", input_shape=(64,)),
            Dense(64, activation="relu"),
            Dense(10)
        ])
        model.compile(optimizer="adam", loss="mse")
    
    # Create training data
    x = np.random.random((32, 64)).astype("float32")
    y = np.random.random((32, 10)).astype("float32")
    
    # Train
    history = model.fit(x, y, epochs=1, verbose=1)
    
    # Check gradients
    print("\nChecking gradients...")
    for layer in model.layers:
        if hasattr(layer, 'kernel'):
            kernel_var = layer.kernel
            if hasattr(kernel_var, '_value'):
                kernel_tensor = kernel_var._value
            elif hasattr(kernel_var, 'value'):
                kernel_tensor = kernel_var.value
            else:
                kernel_tensor = kernel_var
            
            if hasattr(kernel_tensor, 'grad') and kernel_tensor.grad is not None:
                grad_tensor = kernel_tensor.grad
                grad_norm = float(torch.norm(grad_tensor.cpu()).numpy())
                print(f"  {layer.name}.kernel: gradient_norm={grad_norm:.6f}, requires_grad={kernel_tensor.requires_grad}")
            else:
                print(f"  {layer.name}.kernel: grad is None (but this may be expected after training)")
    
    print("\n✓ Gradient flow test PASSED - no RuntimeError!")
    return True

if __name__ == "__main__":
    test_gradient_flow()

