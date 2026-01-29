# TODO: Fix DTensor Sharding in PyTorch Backend

## Objective
Fix the DTensor sharding issue in `keras/src/backend/torch/distribution_lib.py` so that ModelParallel properly shards variables across devices.

## Root Cause Analysis
1. The `needs_sharding` check in `distribute_variable` incorrectly handles tuple layouts
2. DeviceMesh is not properly registered in global state before variable creation
3. Path matching between Keras (`dense/kernel`) and PyTorch (`dense.weight`) formats is incomplete
4. Manual sharding fallback creates parameters without proper DTensor metadata

## Implementation Plan COMPLETED

### Step 1: Fix `distribute_variable` function (High Priority) ✅
- [x] Fix `needs_sharding` logic to correctly identify sharding axes
- [x] Ensure DeviceMesh is properly registered before variable creation
- [x] Add proper DTensor wrapping with correct placements
- [x] Improve error handling and debug logging

### Step 2: Fix `_to_backend_mesh` function ✅
- [x] Ensure the backend mesh is properly stored in global state
- [x] Add proper device handling for multi-GPU setups
- [x] Add debug logging for mesh creation

### Step 3: Add `_get_mesh_info` helper function ✅
- [x] Create a function to retrieve mesh information for debugging
- [x] Store mesh shape and dimension names

### Step 4: Update `ModelParallel.__init__` in distribution_lib.py ✅
- [x] Ensure backend mesh is created when ModelParallel is initialized
- [x] This triggers registration in global state before variable creation

## Changes Made

### `keras/src/backend/torch/distribution_lib.py`

1. **`distribute_variable` function**:
   - Fixed `needs_sharding` logic - now checks `axis is not None` instead of `axis != 'batch'`
   - Added early retrieval of `device_mesh` from global state
   - Added proper DTensor wrapping with `_axis_names_to_placements`
   - Improved debug logging at each step
   - Better fallback handling for when DTensor is not available

2. **`_to_backend_mesh` function**:
   - Added extensive debug logging for mesh creation
   - Improved device index extraction
   - Better handling of CPU vs CUDA devices
   - Clear logging of mesh shape and dimension names

3. **`_get_mesh_info` function** (NEW):
   - Helper function to retrieve mesh information for debugging
   - Returns dict with shape, dim_names, and devices

### `keras/src/distribution/distribution_lib.py`

4. **`ModelParallel.__init__`**:
   - Added code to trigger `device_mesh.backend_mesh` creation
   - This ensures the PyTorch DTensor DeviceMesh is registered in global state
   - Before variables are created in the scope

## Testing
- Run `kaggle_distributed_test.py` to verify the fix
- Check that physical storage verification shows sharded tensors
- Verify gradient flow works correctly
- Look for "DTensor not available" messages - should now use DTensor when available

## Expected Behavior After Fix
1. When DTensor is available and properly initialized:
   - Variables should be wrapped as DTensor Parameters
   - Physical storage verification should show actual sharded shapes
   - Logs should show DTensor creation instead of manual sharding

2. When DTensor is not available:
   - Fallback to manual sharding via Parameter slicing
   - Logs should show "DTensor not available" message

