# Fix Plan: CUDA Tensor to NumPy Conversion Error

## Problem Analysis

The test script `test_torch_model_parallel_2gpu.py` is failing with errors like:
```
can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
```

This occurs because the script is calling `.numpy()` directly on CUDA tensors, which PyTorch doesn't allow. The proper approach is to:
1. Call `.cpu()` on the tensor first to move it to host memory
2. Then call `.numpy()` to convert to numpy array

## Root Cause

The issue is in the variable inspection code in `test_torch_model_parallel_2gpu.py` where it tries to get numpy arrays from CUDA tensors without moving them to CPU first:

```python
# This FAILS on CUDA tensors:
initial_var_values[var_path] = var.value.numpy().copy()
# or
current_value = var.value.numpy()
```

## Solution

Use the `convert_to_numpy()` function from `keras.src.backend.torch.core` which already handles CUDA tensors correctly by:
1. Detaching the tensor from the computation graph
2. Moving it to CPU if needed
3. Converting to numpy

## Files to Modify

### 1. `test_torch_model_parallel_2gpu.py`

**Changes needed in multiple locations:**

#### Location 1: Step 7 - Variable inspection section (around line 220)
```python
# BEFORE (fails):
with model_parallel.scope():
    var_layout = model_parallel.get_variable_layout(var)

# AFTER (add import at top of section):
from keras.src.backend.torch.core import convert_to_numpy
```

#### Location 2: Step 15 - Capturing initial variable values (around line 560)
```python
# BEFORE:
initial_var_values[var_path] = var.value.numpy().copy()
elif hasattr(var, 'numpy'):
    initial_var_values[var_path] = var.numpy().copy()

# AFTER:
from keras.src.backend.torch.core import convert_to_numpy
try:
    if hasattr(var, 'value') and hasattr(var.value, 'numpy'):
        initial_var_values[var_path] = convert_to_numpy(var.value).copy()
    elif hasattr(var, 'numpy'):
        initial_var_values[var_path] = convert_to_numpy(var).copy()
except Exception as e:
    initial_var_values[var_path] = None
```

#### Location 3: Step 15 - Checking if variables changed (around line 600)
```python
# BEFORE:
current_value = var.value.numpy()
# or
current_value = var.numpy()

# AFTER:
current_value = convert_to_numpy(var.value)
# or
current_value = convert_to_numpy(var)
```

## Implementation Steps

1. Add `from keras.src.backend.torch.core import convert_to_numpy` import at the top of the relevant sections
2. Replace all direct `.numpy()` calls on variables with `convert_to_numpy()` calls
3. Add proper error handling for the conversion

## Testing

After fixing the test script, run:
```bash
CUDA_VISIBLE_DEVICES=0,1 python test_torch_model_parallel_2gpu.py
```

Expected results:
- No CUDA tensor to numpy conversion errors
- Training completes successfully
- Variables can be properly inspected before and after training

