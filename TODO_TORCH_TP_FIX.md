# Tensor Parallelism Torch Backend Fix Plan

## Issue Analysis
The Torch distribution library needs to be aligned with JAX's interface for tensor parallelism to work correctly.

## Key Differences to Fix

### 1. `all_gather` function signature
- **JAX**: `all_gather(x, axis, axis_name="model")` - axis is positional
- **Torch**: `all_gather(tensor, axis=0, axis_name="model")` - axis needs to be first positional arg
- **Fix**: Change signature to `all_gather(tensor, axis=0, axis_name="model")`

### 2. `all_reduce` function
- Already has `axis_name` parameter - verify it works correctly

### 3. Missing utility functions
- Add `is_distributed_initialized()`
- Add `get_local_rank()`
- Add `get_local_world_size()`
- Add convenience functions: `tensor_parallel_all_gather`, `model_parallel_all_reduce`, `data_parallel_all_reduce`

### 4. `_get_current_rank` and `_get_current_device`
- Need to ensure these are available and work correctly in distributed setting

## Files to Modify
1. `keras/src/backend/torch/distribution_lib.py` - Fix function signatures and add missing utilities

## Testing
- Run existing tests in `test_torch_distribution_fix.py`
- Verify compatibility with `autoconfig.py` patterns

