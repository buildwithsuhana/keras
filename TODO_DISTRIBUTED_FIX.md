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

## Issue 3: DTensor Mixed Tensor Error - SOLUTION: Use parallelize_module
**Problem:** 
```
RuntimeError: aten.mm.default: got mixed torch.Tensor and DTensor
RuntimeError: aten.sub.Tensor: got mixed torch.Tensor and DTensor
```

### Root Cause:
When kernel is a DTensor (due to model parallel sharding) but input is a regular tensor, PyTorch's DTensor operations fail because they require both operands to be DTensors.

### Solution Applied - Use `torch.distributed.tensor.parallel.parallelize_module`:
Instead of manually converting tensors to DTensors in numpy.py operations, we use PyTorch's built-in `parallelize_module` with `ColwiseParallel`/`RowwiseParallel`. This approach:

1. **Automatically handles DTensor conversions** at the module level
2. **No need to modify individual operations** (matmul, add, subtract, multiply)
3. **Cleaner architecture** - PyTorch handles all the complexity

### Implementation:
Added to `keras/src/backend/torch/distribution_lib.py`:
- `parallelize_torch_module()` - Wrapper for `parallelize_module()`
- `create_tp_plan_from_layout_map()` - Translates Keras specs to PyTorch parallel styles

## Implementation Steps
- [x] Analyze error and understand root cause
- [x] Modify `distribute_variable` to handle non-floating point tensors
- [x] Add debug logging for tensor shapes
- [x] Fix cache key in `_to_backend_mesh` for rank synchronization
- [x] Add `parallelize_torch_module()` helper function
- [x] Add `create_tp_plan_from_layout_map()` helper function
- [x] Revert numpy.py changes (no longer needed with parallelize_module approach)

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

