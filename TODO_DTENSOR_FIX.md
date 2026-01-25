# DTensor Fix Progress Tracker

## Task
Fix the `RuntimeError: aten.sub.Tensor: got mixed torch.Tensor and DTensor` error when using Keras ModelParallel with Torch backend.

## Root Cause
When `torch.compile`/`dynamo` is enabled, it intercepts operations at the PyTorch dispatch level and calls `aten.sub.Tensor` directly, bypassing our Keras wrapper functions. The DTensor's `__torch_dispatch__` mechanism doesn't automatically convert regular tensors to DTensors when mixed in operations.

## Solution Implemented

### 1. Monkey-patched DTensor.__torch_dispatch__
Instead of trying to subclass DTensor (which has a complex `__new__` method), we monkey-patch the DTensor class's `__torch_dispatch__` method when the distribution_lib module is imported. This ensures that all DTensor operations automatically handle mixed DTensor/torch.Tensor operands.

### 2. Updated numpy.py functions
Added DTensor handling to:
- `subtract` - PRIMARY FIX (causes the current error)
- `multiply` - For consistency
- `minimum` - For consistency
- `maximum` - For consistency
- `divide` - For consistency

## Files Modified
- `/Users/suhanaaa/keras/keras/src/backend/torch/distribution_lib.py`
- `/Users/suhanaaa/keras/keras/src/backend/torch/numpy.py`

## How the Monkey-Patch Works
When the distribution_lib module is imported:
1. `_patch_dtensor_for_mixed_operations()` is called
2. It stores the original `DTensor.__torch_dispatch__` method
3. It replaces it with a patched version that:
   - Checks if any argument is a DTensor
   - If yes, converts all regular torch.Tensor operands to DTensors with compatible mesh/placements
   - Calls the original operation with converted operands

This approach is applied globally to all DTensor operations, ensuring consistent behavior even when torch.compile/dynamo intercepts operations.

## Progress
- [x] Fix `subtract` function in numpy.py
- [x] Fix `multiply` function in numpy.py
- [x] Fix `minimum` function in numpy.py
- [x] Fix `maximum` function in numpy.py
- [x] Fix `divide` function in numpy.py
- [x] Monkey-patch DTensor.__torch_dispatch__ in distribution_lib.py

## Summary
By monkey-patching the DTensor class's `__torch_dispatch__` method, all distributed tensor operations now automatically handle mixed DTensor/torch.Tensor operations. When torch.compile/dynamo intercepts operations at the dispatch level, the patched handler converts regular tensors to DTensors before performing the operation, preventing the "mixed torch.Tensor and DTensor" error.

