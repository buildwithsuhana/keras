# Todo List: Fix Tensor Parallel 2D DeviceMesh Error

## Problem
PyTorch's `parallelize_module` API only accepts a 1D DeviceMesh, but the code creates a 2D DeviceMesh with shape `(1, 2)` for model parallelism.

## Error
```
ValueError: Tensor Parallel only accepts a 1D DeviceMesh, but found 2D!
If you have a 2-D or N-D device_mesh, consider passing in device_mesh["tp"]
```

## Solution
Modify `keras/src/backend/torch/distribution_lib.py`:
1. In `parallelize_torch_module` function, extract 1D mesh from 2D DeviceMesh for TP operations
2. Create helper function to get 1D TP mesh from 2D DeviceMesh

## Files Modified
- `keras/src/backend/torch/distribution_lib.py`

## Changes Made
1. [x] Added helper function `_get_tp_mesh_from_2d_mesh` to extract 1D TP mesh
2. [x] Modified `parallelize_torch_module` to handle 2D DeviceMesh

## Summary
The fix extracts the "model" axis from a 2D DeviceMesh (shape=(1, 2)) and creates a 1D DeviceMesh for tensor parallel operations. This allows PyTorch's `parallelize_module` to work correctly with Keras's 2D DeviceMesh configurations.

