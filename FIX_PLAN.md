# Fix Plan: Disable Manual Sharding for ModelParallel

## Problem
When using `ModelParallel` distribution, manual sharding still happens because:
1. `Variable._layout` is not being set properly before `_distribute_parameter` is called
2. `distribute_variable` receives `layout=None` and falls back to regular Parameters
3. This conflicts with the `parallelize_module` approach which should handle DTensor conversion

## Root Cause Analysis

### Issue 1: Variable._initialize_layout() not capturing layout correctly
In `core.py`, the `Variable._initialize_layout()` method checks `distribution()` but may not properly set `self._layout` when the layout is returned.

### Issue 2: distribute_variable called with layout=None
When `Variable._distribute_parameter()` is called without a layout, it passes `layout=None` to `distribute_variable`, which then creates regular Parameters.

### Issue 3: ModelParallel detection in distribute_variable
The `distribute_variable` function checks `if layout is None` before checking if ModelParallel is active, causing it to skip DTensor creation.

## Solution

### Step 1: Fix Variable._initialize_layout() in core.py
- Ensure the layout is properly captured from the distribution context
- Set `self._layout` when `tensor_layout` is not None

### Step 2: Fix distribute_variable in torch/distribution_lib.py
- Check if ModelParallel is active even when layout is None
- If ModelParallel is active, use the distribution's layout_map to get the proper layout

### Step 3: Ensure parallelize_module path is used
- When ModelParallel is active, `distribute_variable` should NOT create DTensor weights
- Instead, it should create regular Parameters and let `parallelize_module` handle the conversion
- This prevents double-sharding issues

## Files to Modify

### 1. keras/src/backend/torch/core.py
- `Variable._initialize_layout()` - Fix layout capture
- `Variable._distribute_parameter()` - Ensure layout is passed correctly

### 2. keras/src/backend/torch/distribution_lib.py
- `distribute_variable()` - Fix ModelParallel detection logic
- Add helper to get layout from distribution when not explicitly provided

## Implementation Details

### core.py changes:
```python
def _initialize_layout(self):
    # Get layout from distribution context if not explicitly set
    if self._layout is None:
        from keras.src.distribution import distribution
        
        dist = distribution()
        if dist is not None:
            tensor_layout = dist.get_variable_layout(self)
            if tensor_layout is not None:
                # Set self._layout from the tensor_layout
                if hasattr(tensor_layout, 'axes'):
                    self._layout = tensor_layout.axes
                elif hasattr(tensor_layout, 'backend_layout'):
                    self._layout = tensor_layout.backend_layout
                else:
                    self._layout = tensor_layout
```

### distribution_lib.py changes:
```python
def distribute_variable(tensor, layout=None, module_name=None):
    # ... existing code ...
    
    # Check if ModelParallel distribution is active
    is_model_parallel = False
    current_distribution = distribution()
    if current_distribution is not None:
        from keras.src.distribution.distribution_lib import ModelParallel
        is_model_parallel = isinstance(current_distribution, ModelParallel)
    
    # If ModelParallel is active and no layout provided, get from distribution
    if is_model_parallel and layout is None:
        layout = current_distribution.get_variable_layout(variable)
    
    # ... rest of function ...
```

## Testing

After implementing the fix, run:
```bash
# Single process test
python kaggle_distributed_test.py

# Multi-process test
torchrun --nproc_per_node=2 kaggle_distributed_test.py
```

Expected behavior:
1. DataParallel: Variables should be replicated (not sharded)
2. ModelParallel: Variables should be created as regular Parameters, then sharded by parallelize_module
3. No "mixed torch.Tensor and DTensor" errors
4. Training should complete successfully

## Follow-up
- Verify the parallelize_module is called during fit()
- Ensure auto-parallelization is triggered at the right time
- Check that gradient computation works correctly

