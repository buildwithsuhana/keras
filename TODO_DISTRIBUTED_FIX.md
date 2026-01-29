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

## Issue 3: DTensor Mixed Tensor Error (Fixed)
**Problem:** 
```
RuntimeError: aten.mm.default: got mixed torch.Tensor and DTensor, 
need to convert all torch.Tensor to DTensor before calling distributed operators!
```

### Root Cause:
When the kernel is a DTensor (due to model parallel sharding) but the input is a regular tensor, PyTorch's DTensor operations fail because they require both operands to be DTensors.

### Fix Applied:
Modified `matmul` function in `keras/src/backend/torch/numpy.py` to:
1. Import `DTensor` and `Replicate` at module level
2. Check if either operand is a DTensor
3. Convert the other operand to a replicated DTensor if needed

## Implementation Steps
- [x] Analyze error and understand root cause
- [x] Modify `distribute_variable` to handle non-floating point tensors
- [x] Add debug logging for tensor shapes
- [x] Fix cache key in `_to_backend_mesh` for rank synchronization
- [x] Fix `matmul` to handle DTensor mixed tensor error
- [x] Test the fix - All tests PASSED

## Test Results
- ✓ Device Detection: PASSED
- ✓ DataParallel: PASSED
- ✓ ModelParallel: PASSED
- ✓ Gradient Flow: PASSED

## Physical Storage Verification (ModelParallel)
Both ranks now correctly shard model weights:
- Layer 0 (dense_3): local_shape=(128, 256) on both Rank 0 and Rank 1
- Layer 1 (dense_4): local_shape=(512, 128) on both Rank 0 and Rank 1
- Layer 2 (dense_5): local_shape=(256, 64) on both Rank 0 and Rank 1
- Layer 3 (dense_6): local_shape=(128, 5) on both Rank 0 and Rank 1

