"""Test script to verify the fix for the RuntimeError: Only Tensors of floating point and complex dtype can require gradients"""

import os
os.environ["KERAS_BACKEND"] = "torch"

import torch
import torch.nn as nn

# Test 1: Test _make_torch_param with various dtypes
print("Test 1: Testing _make_torch_param with various dtypes...")

from keras.src.backend.torch.layer import _make_torch_param, DEBUG_LAYER

# Enable debug mode
os.environ["KERAS_TORCH_LAYER_DEBUG"] = "1"

# Test with float tensor
float_tensor = torch.randn(3, 3)
result = _make_torch_param(float_tensor)
assert isinstance(result, nn.Parameter), "Float tensor should be wrapped as Parameter"
assert result.requires_grad == True, "Float tensor should have requires_grad=True"
print("  ✓ Float tensor: OK")

# Test with non-float tensor (int32)
int_tensor = torch.ones(3, 3, dtype=torch.int32)
result = _make_torch_param(int_tensor)
assert not isinstance(result, nn.Parameter), "Int tensor should NOT be wrapped as Parameter"
assert isinstance(result, torch.Tensor), "Int tensor should remain as Tensor"
print("  ✓ Int32 tensor: OK")

# Test with existing Parameter
existing_param = nn.Parameter(torch.randn(3, 3))
result = _make_torch_param(existing_param)
assert result is existing_param, "Existing Parameter should be returned as-is"
print("  ✓ Existing Parameter: OK")

# Test 2: Test TorchLayer with non-float dtypes
print("\nTest 2: Testing TorchLayer with non-float dtypes...")

from keras.src.backend.torch.layer import TorchLayer
from keras.src.layers import Layer

class TestLayer(TorchLayer):
    def __init__(self):
        super().__init__()
        # Add a non-float dtype variable (like int32)
        self.non_float_var = self.add_weight(
            shape=(3, 3),
            dtype="int32",
            initializer="zeros",
            trainable=True
        )
        # Add a float dtype variable
        self.float_var = self.add_weight(
            shape=(3, 3),
            dtype="float32",
            initializer="ones",
            trainable=True
        )

# Build the layer
layer = TestLayer()
_ = layer.build((3, 3))

# Check that torch_params contains both variables
assert "non_float_var" in layer.torch_params, "non_float_var should be in torch_params"
assert "float_var" in layer.torch_params, "float_var should be in torch_params"

# Check that non_float_var is NOT a Parameter
assert not isinstance(layer.torch_params["non_float_var"], nn.Parameter), \
    "non_float_var should NOT be a Parameter (int32 dtype)"

# Check that float_var IS a Parameter
assert isinstance(layer.torch_params["float_var"], nn.Parameter), \
    "float_var should be a Parameter (float32 dtype)"

print("  ✓ Non-float variable correctly NOT wrapped as Parameter")
print("  ✓ Float variable correctly wrapped as Parameter")

# Test 3: Test that named_parameters works correctly
print("\nTest 3: Testing named_parameters...")

params = list(layer.named_parameters())
param_names = [name for name, _ in params]

assert "float_var" in param_names, "float_var should be in named_parameters"
assert "non_float_var" not in param_names, "non_float_var should NOT be in named_parameters (int32 dtype)"

print("  ✓ named_parameters correctly excludes non-float parameters")

print("\n" + "="*50)
print("All tests passed! The fix is working correctly.")
print("="*50)

