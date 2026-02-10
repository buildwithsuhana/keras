# TODO: Fix OPT Model Parallel Testing Issues

## Issues Identified

1. **Mixed DTensor Error**: `RuntimeError: got mixed torch.Tensor and DTensor, need to convert all torch.Tensor to DTensor before calling distributed operators!`
   - Occurs in ModelParallel when embedding layer weights are DTensors but inputs are regular tensors

2. **Non-floating point dtype error**: `RuntimeError: Only Tensors of floating point and complex dtype can require gradients`
   - Happens when int32 tensors (like embedding indices) are wrapped with torch.nn.Parameter

## Fix Plan

### Step 1: Update `embedding.py`
- Override `call()` method to detect DTensor weights
- Convert regular tensor inputs to DTensors when weights are DTensors
- **Status**: ✅ COMPLETED

### Step 2: Update `distribution_lib.py`
- Fix `distribute_variable()` to NOT wrap non-floating point tensors (int32) with Parameter
- Add helper function to convert inputs to DTensors when weight is DTensor
- **Status**: ✅ COMPLETED

### Step 3: Test the fix
- Run opt_simple_test.py with ModelParallel
- Verify both DataParallel and ModelParallel tests pass
- **Status**: Pending - User needs to test

## Files Modified

1. `keras/src/layers/core/embedding.py`
2. `keras/src/backend/torch/distribution_lib.py`

## Summary of Changes

### embedding.py
- Added DTensor handling in `call()` method
- When embedding weights are DTensors, converts inputs to DTensors
- Uses local embedding tensor for the embedding lookup operation

### distribution_lib.py
- Added `is_integer_dtype` check in `distribute_variable()`
- When creating sharded DTensors, returns DTensor directly for integer tensors (no Parameter wrapper)
- Added debug logging for integer tensor handling

## Testing Command

```bash
# Multi-GPU (2 GPUs):
torchrun --nproc_per_node=2 opt_simple_test.py

# Single GPU:
python opt_simple_test.py
```


