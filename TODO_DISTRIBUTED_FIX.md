# TODO: Distributed Training Fix - COMPLETED

## Issue 1: RuntimeError for Non-Floating Point Tensors
```
RuntimeError: Only Tensors of floating point and complex dtype can require gradients
```

### Root Cause:
The `distribute_variable` function in `keras/src/backend/torch/distribution_lib.py` wraps ALL tensors in `torch.nn.Parameter`, but PyTorch requires floating-point tensors for parameters that track gradients. The `iterations` variable in the optimizer has `dtype="int"` and `trainable=False`.

### Fix Applied:
Modified `distribute_variable` to check if tensor is floating-point before wrapping in Parameter.

## Issue 2: Rank Desync in ModelParallel (Fixed)
**Problem:** Rank 1 was using stale 1D mesh from DataParallel test when running ModelParallel test.

### Root Cause:
The caching logic in `_to_backend_mesh` was using object `id()` which wasn't unique enough across different mesh configurations.

### Fix Applied:
Changed cache key from `f"torch_device_mesh_{id(device_mesh)}"` to configuration-based key:
```python
shape_str = str(device_mesh.shape)
axes_str = str(device_mesh.axis_names)
cache_key = f"torch_mesh_{shape_str}_{axes_str}"
```

This ensures:
- DP with shape=(2,) and axes=['batch'] → key: `torch_mesh_(2,)_['batch']`
- MP with shape=(1, 2) and axes=['batch', 'model'] → key: `torch_mesh_(1, 2)_['batch', 'model']`

## Implementation Steps
- [x] Analyze error and understand root cause
- [x] Modify `distribute_variable` to handle non-floating point tensors
- [x] Add debug logging for tensor shapes
- [x] Fix cache key in `_to_backend_mesh` for rank synchronization
- [x] Test the fix - All tests PASSED

## Test Results
- ✓ Device Detection: PASSED
- ✓ DataParallel: PASSED
- ✓ ModelParallel: PASSED
- ✓ Gradient Flow: PASSED

