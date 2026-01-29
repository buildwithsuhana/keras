# Fix Plan: ModelParallel and DataParallel DTensor Sharding

## Problem Analysis

From the test log, we see:
```
DEBUG | [Rank 00] distribute_variable: shape=torch.Size([128, 512]), ... layout=(None, 'model'), world_size=2
DEBUG | [Rank 00] distribute_variable: current_distribution=None, layout is None=False
DEBUG | [Rank 00] No sharding needed, replicating: shape=torch.Size([128, 512]), dtype=torch.float32
```

The issue is that `distribution()` returns `None` even though we're inside the distribution scope.

## Root Cause

The `distribution()` function retrieves the distribution from global state using `GLOBAL_ATTRIBUTE_NAME = "distribution"`. The issue is that:
1. `set_distribution()` sets the distribution in global state
2. But the global state might not be properly shared between modules
3. Or the scope context manager isn't working as expected

## Information Gathered

1. **distribution_lib.py** - High-level APIs
   - `distribution()` - retrieves from `global_state.get_global_attribute(GLOBAL_ATTRIBUTE_NAME)`
   - `set_distribution(value)` - sets with `global_state.set_global_attribute(GLOBAL_ATTRIBUTE_NAME, value)`
   - `Distribution.scope()` - calls `set_distribution(self)` in the context

2. **torch/distribution_lib.py** - Backend-specific
   - `distribute_variable()` - calls `from keras.src.distribution.distribution_lib import distribution`
   - This import gets the high-level `distribution()` function
   - But it might be importing a different module instance

## Plan

### Step 1: Add detailed debug logging to trace the distribution flow
- [x] Already added debug logging to `distribution()`, `set_distribution()`, `distribute_variable()`, `_to_backend_mesh()`
- [x] Created `debug_distribution.py` script to test

### Step 2: Verify global state is properly shared between modules
- Check if `keras.src.distribution.distribution_lib` and `keras.distribution` use the same module
- The import `from keras.src.distribution.distribution_lib import distribution` should get the correct function

### Step 3: Fix the distribution lookup if needed
- If `distribution()` returns `None`, we need to find why
- The issue might be with how global_state is being accessed

### Step 4: Implement the actual sharding fix
- Once distribution is properly accessible, ensure DTensor sharding works
- Verify mesh_dim_names matches the layout axes

## Expected Debug Output

When running with debug mode, we should see:
1. `set_distribution(ModelParallel(...))` called when entering scope
2. `distribution()` returns the ModelParallel instance inside the scope
3. `distribute_variable` sees the distribution and applies sharding
4. `_to_backend_mesh` creates the DTensor DeviceMesh with correct mesh_dim_names
5. Sharding placements are correctly calculated

## Files to Modify

1. `keras/src/backend/torch/distribution_lib.py` - Added debug logging
2. `keras/src/distribution/distribution_lib.py` - Added debug logging
3. `keras/src/backend/torch/core.py` - May need modifications for proper layout handling

