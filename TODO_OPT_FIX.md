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
- **Status**: ✅ COMPLETED

### Step 3: Update `numpy.py`
- Fix `take()` function to handle DTensor case for embedding lookup
- When x is DTensor and indices is not, convert indices to DTensor
- **Status**: ✅ COMPLETED

### Step 4: Test the fix
- Run opt_simple_test.py with ModelParallel
- Verify both DataParallel and ModelParallel tests pass
- **Status**: Pending - User needs to test

## Files Modified

1. `keras/src/layers/core/embedding.py`
2. `keras/src/backend/torch/distribution_lib.py`
3. `keras/src/backend/torch/numpy.py`

## Summary of Changes

### embedding.py
- Added DTensor handling in `call()` method
- When embedding weights are DTensors, converts inputs to DTensors
- Uses local embedding tensor for the embedding lookup operation

### distribution_lib.py
- Added `is_integer_dtype` check in `distribute_variable()`
- When creating sharded DTensors, returns DTensor directly for integer tensors (no Parameter wrapper)
- Added debug logging for integer tensor handling

### numpy.py
- Fixed `take()` function to handle DTensor embedding lookup
- When x (embeddings) is a DTensor and indices is not, converts indices to DTensor
- This prevents "mixed torch.Tensor and DTensor" error

## Testing Command

```bash
# Multi-GPU (2 GPUs):
torchrun --nproc_per_node=2 opt_simple_test.py

# Single GPU:
python opt_simple_test.py
```


