# Plan: Fix Sharding Not Happening in PyTorch Backend

## Information Gathered

Based on the analysis of the codebase and the error logs, here's what we found:

### Root Causes

1. **torch.distributed not initialized**: The test script never calls `distribution_lib.initialize()` before creating the model. This causes `_check_distributed_initialized()` to return `False`, which skips the actual DTensor sharding.

2. **Keras DeviceMesh ≠ PyTorch DeviceMesh**: The logs show "✓ Created DeviceMesh: <DeviceMesh shape=(2,), axis_names=['model']>" - this is the Keras-level DeviceMesh, NOT the PyTorch DTensor DeviceMesh needed for physical sharding.

3. **Simulated CPU devices don't work with DTensor**: The test uses simulated CPU devices ("cpu:0", "cpu:1"), but PyTorch DTensor's `init_device_mesh()` requires actual compute devices (CUDA GPUs). On CPU-only systems, DTensor can't work.

4. **Layout assignment happens but not physical sharding**: The logs show "[✗ SHARDED axes=('model', None)]" - meaning Keras layout assignment works (variables have their layouts set), but physical DTensor sharding fails.

### Key Code Locations

- **Variable initialization**: `keras/src/backend/torch/core.py` - `_initialize()` method
- **DTensor distribution**: `keras/src/backend/torch/distribution_lib.py` - `distribute_variable()` and `_get_default_device_mesh()`
- **Layout assignment**: `keras/src/distribution/distribution_lib.py` - `ModelParallel.get_variable_layout()`

## Plan

### Step 1: Fix Variable Initialization to Enable Sharding

**File**: `keras/src/backend/torch/core.py`

**Changes**:
1. Remove the dependency on `_check_distributed_initialized()` for distributing variables
2. Call `initialize()` if not already done (auto-initialize)
3. Try to create DTensor even if distributed isn't fully initialized
4. Add proper fallback handling when DTensor fails

### Step 2: Improve Distribution Library Auto-Initialization

**File**: `keras/src/backend/torch/distribution_lib.py`

**Changes**:
1. Make `initialize()` more robust for single-process/CPU scenarios
2. Create a fallback CPU-based DeviceMesh that can work without full distributed initialization
3. Add better error messages for debugging
4. Fix `_get_default_device_mesh()` to work with the test's simulated devices

### Step 3: Update Test Script to Properly Initialize Distribution

**File**: `test_torch_model_parallel_opt125m_fixed.py`

**Changes**:
1. Call `distribution_lib.initialize()` before creating the model
2. Pass proper device information to the initialization
3. Add debug output to verify DTensor creation

## Implementation Details

### Key Changes to torch/core.py

The current code:
```python
# Apply distribution sharding if available
distribution = global_state.get_global_attribute("distribution")
if distribution is not None:
    try:
        from keras.src.distribution import TensorLayout
        tensor_layout = distribution.get_variable_layout(self)
        if tensor_layout is not None and isinstance(tensor_layout, TensorLayout):
            # Use distribution_lib to distribute the tensor
            from keras.src.backend.torch import distribution_lib
            if distribution_lib._check_distributed_initialized():
                tensor_value = distribution_lib.distribute_variable(
                    tensor_value, 
                    tensor_layout, 
                    None
                )
    except Exception:
        pass  # Fall back to regular tensor
```

Should be changed to:
```python
# Apply distribution sharding if available
distribution = global_state.get_global_attribute("distribution")
if distribution is not None:
    try:
        from keras.src.distribution import TensorLayout
        tensor_layout = distribution.get_variable_layout(self)
        if tensor_layout is not None and isinstance(tensor_layout, TensorLayout):
            # Use distribution_lib to distribute the tensor
            from keras.src.backend.torch import distribution_lib
            # Try to initialize if not already done
            if not distribution_lib._check_distributed_initialized():
                try:
                    distribution_lib.initialize()
                except Exception:
                    pass
            # Try to distribute the variable
            tensor_value = distribution_lib.distribute_variable(
                tensor_value, 
                tensor_layout, 
                None
            )
    except Exception:
        pass  # Fall back to regular tensor
```

### Key Changes to distribution_lib.py

1. Fix `_get_default_device_mesh()` to work with CPU devices properly
2. Create a logical mesh for single-process scenarios
3. Add fallback for when real DTensor can't be created

## Followup Steps

After implementing the fix:
1. Run the test script to verify sharding happens
2. Check that DTensor parameters are created
3. Verify variable shapes are actually sharded
4. Test training to ensure it works end-to-end

