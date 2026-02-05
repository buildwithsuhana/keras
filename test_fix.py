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

from keras.src.layers import Layer

class TestLayer(Layer):
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

# Print the keys to debug
print(f"torch_params keys: {list(layer.torch_params.keys())}")

# Check that torch_params contains both variables
# Note: Variable paths include layer name as prefix when auto-named
assert any("variable" in key for key in layer.torch_params.keys()), \
    "Variables should be in torch_params"

# Since we have two variables, verify they exist
assert len(layer.torch_params) == 2, f"Expected 2 variables, got {len(layer.torch_params)}"

# Check that float and non-float variables are handled correctly
float_found = False
non_float_found = False
for key, value in layer.torch_params.items():
    if hasattr(value, 'dtype'):
        if value.dtype.is_floating_point or value.dtype.is_complex:
            # Float variable should be a Parameter
            assert isinstance(value, nn.Parameter), \
                f"Float variable at key '{key}' should be a Parameter, got {type(value)}"
            float_found = True
            print(f"  ✓ Float variable at '{key}' correctly wrapped as Parameter")
        else:
            # Non-float variable should NOT be a Parameter
            assert not isinstance(value, nn.Parameter), \
                f"Non-float variable at key '{key}' should NOT be a Parameter"
            non_float_found = True
            print(f"  ✓ Non-float variable at '{key}' correctly NOT wrapped as Parameter")

assert float_found, "Float variable should be found"
assert non_float_found, "Non-float variable should be found"

# Test 3: Test that named_parameters works correctly
print("\nTest 3: Testing named_parameters...")

params = list(layer.named_parameters())
param_names = [name for name, _ in params]

print(f"  named_parameters: {param_names}")

# Only the float variable should be in named_parameters (not the int32 one)
assert len(param_names) == 1, \
    f"Only float variable should be in named_parameters. Got: {param_names}"

print("  ✓ named_parameters correctly excludes non-float parameters")

print("\n" + "="*50)
print("All tests passed! The fix is working correctly.")
print("="*50)

