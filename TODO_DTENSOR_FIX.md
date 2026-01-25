# DTensor Fix Progress Tracker

## Task
Fix the `RuntimeError: aten.sub.Tensor: got mixed torch.Tensor and DTensor` error when using Keras ModelParallel with Torch backend.

## Root Cause
When `torch.compile`/`dynamo` is enabled, it intercepts operations at the PyTorch dispatch level and calls `aten.sub.Tensor` directly, bypassing our Keras wrapper functions. The DTensor's `__torch_dispatch__` mechanism doesn't automatically convert regular tensors to DTensors when mixed in operations.

The error occurs because:
1. Model weights are DTensor (sharded) due to ModelParallel
2. Input data (`x_train`, `y_train`) are regular `torch.Tensor`
3. During training with `torch.compile`, PyTorch's internal operations are called directly
4. DTensor operations require ALL operands to be DTensors

## Solution Implemented

### 1. Updated `distribute_data_input` in torch/distribution_lib.py
- Added `batch_dim_name` parameter to handle batch dimension sharding
- Properly converts input data to DTensor format with compatible mesh and placements
- Handles edge cases: DTensor input (return as-is), no layout (return as-is), etc.

### 2. Updated `_distribute_data` in trainers/epoch_iterator.py
- Now passes `batch_dim_name` from the distribution to `distribute_data_input`
- Ensures input data is properly distributed when using ModelParallel

### 3. DTensor handling in numpy.py functions (already implemented)
- Added DTensor handling to: `subtract`, `add`, `multiply`, `minimum`, `maximum`, `divide`, `matmul`
- Each function now checks if one operand is a DTensor and converts regular tensors to DTensors

## Files Modified
- `/Users/suhanaaa/keras/keras/src/backend/torch/distribution_lib.py` - Enhanced `distribute_data_input`
- `/Users/suhanaaa/keras/keras/src/trainers/epoch_iterator.py` - Pass `batch_dim_name` to `distribute_data_input`
- `/Users/suhanaaa/keras/keras/src/backend/torch/numpy.py` - DTensor handling (already present)

## Progress
- [x] Fix `subtract` function in numpy.py
- [x] Fix `add` function in numpy.py
- [x] Fix `multiply` function in numpy.py
- [x] Fix `minimum` function in numpy.py
- [x] Fix `maximum` function in numpy.py
- [x] Fix `divide` function in numpy.py
- [x] Fix `matmul` function in numpy.py
- [x] Enhance `distribute_data_input` in distribution_lib.py
- [x] Update `_distribute_data` in epoch_iterator.py to pass batch_dim_name

## Summary
The input data distribution system now properly converts input data to DTensor format. When using ModelParallel with Torch backend, input data batches are automatically converted to DTensors with compatible mesh and placements, ensuring that all operations during training have consistent tensor types. This prevents the "mixed torch.Tensor and DTensor" error when torch.compile is active.

