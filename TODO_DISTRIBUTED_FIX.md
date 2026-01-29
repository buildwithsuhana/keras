# TODO: Distributed Training Fix

## Error Analysis
```
RuntimeError: Only Tensors of floating point and complex dtype can require gradients
```

**Root Cause:** The `distribute_variable` function in `keras/src/backend/torch/distribution_lib.py` wraps ALL tensors in `torch.nn.Parameter`, but PyTorch requires floating-point tensors for parameters that track gradients. The `iterations` variable in the optimizer has `dtype="int"` and `trainable=False`, but it's still being wrapped in a `torch.nn.Parameter`.

## Fix Applied

### 1. Modified `distribute_variable` in `keras/src/backend/torch/distribution_lib.py`
- Check if the tensor is floating-point or complex before wrapping in `torch.nn.Parameter`
- Non-floating point tensors (like integer iterations counter) are returned as regular tensors
- Added debug logging to show tensor shapes on each rank

### Key Changes:
- Convert tensor first and check `is_floating_point` or `is_complex` dtype
- Return non-floating point tensors as-is without wrapping in `torch.nn.Parameter`
- Added `[Rank XX]` prefix to debug logs for distributed context
- Added logging for distributed tensor shapes after sharding

## Implementation Steps
- [x] Analyze error and understand root cause
- [x] Modify `distribute_variable` to handle non-floating point tensors
- [x] Add debug logging for tensor shapes
- [ ] Test the fix
- [ ] Verify the fix resolves the RuntimeError
- [ ] Test with both DataParallel and ModelParallel scenarios

