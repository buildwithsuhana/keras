# TODO: Fix OPT Model Parallelism Sharding

## Problem Summary
The OPT model with hybrid Data Parallel + Model Parallel has issues:
1. Sharding not properly applied - weights show "Sharded: 0"
2. Runtime error during forward pass: "The size of tensor a (6) must match the size of tensor b (12)"
3. Pattern mismatch: `.*feed_forward.*` doesn't match `feedforward` (underscore issue)
4. NCCL "Duplicate GPU detected" - both ranks using same GPU
5. Missing module: `ModuleNotFoundError: No module named 'torch.distributed.tensor.api'`
6. `distribution()` returns `None` during forward pass
7. **TypeError: 'TensorLayout' object is not iterable** - Fixed in V12

## Root Causes Identified

### 1. Pattern Mismatch (FIXED in V11)
- Pattern: `".*feed_forward.*kernel"` 
- Actual weight name: `"transformer_layer_0/feedforward_intermediate_dense/kernel"`
- Issue: underscore vs no underscore
- Fix: Changed pattern to `".*feedforward.*kernel"` (no underscore)

### 2. Wrong Dimension for Sharding (FIXED in V10)
- In `_layout_to_placements()`:
  - Layout `(None, 'model')` should shard on tensor dimension 1
  - Original code returned `[Shard(0)]` (mesh dimension)
  - Fixed to return `[Shard(1)]` (tensor dimension where 'model' appears)

### 3. NCCL "Duplicate GPU detected" Error (FIXED in V11)
- Both ranks trying to use same GPU because proper CUDA device assignment wasn't working
- Fix: Set CUDA device BEFORE any torch.cuda calls or distributed init
- Each rank uses unique GPU: `gpu_id = local_rank % num_gpus`

### 4. Missing `torch.distributed.tensor.api` Module (FIXED)
- Wrong import path used in some places
- Correct import: `from torch.distributed._tensor.api import distribute_tensor as torch_distribute_tensor`

### 5. `distribution()` returns `None` (FIXED in V11)
- Distribution context not properly set during forward pass
- Fix: Ensure `set_distribution()` is called in the test script before redistribution

### 6. TypeError: 'TensorLayout' object is not iterable (FIXED in V12)
- The `_layout_to_placements()` function tried to iterate directly over `TensorLayout` objects
- Fix: Check if layout is a `TensorLayout` instance and extract `.axes` before iterating
- Similar fix applied to `distribute_tensor()` and `distribute_variable()` functions

## Changes Made in V12

### File: keras/src/backend/torch/distribution_lib.py

#### Fix 1: `_layout_to_placements()` function
```python
def _layout_to_placements(layout, tensor, device_mesh):
    # ...
    # CRITICAL FIX: Handle TensorLayout objects properly
    from keras.src.distribution.distribution_lib import TensorLayout
    if isinstance(layout, TensorLayout):
        # Get the axes tuple from the TensorLayout object
        layout_axes = layout.axes
    else:
        layout_axes = layout
    
    # Now use layout_axes for iteration instead of layout
    if layout_axes is not None:
        for i, axis in enumerate(layout_axes):
            if axis == 'model':
                return [Shard(i)]
```

#### Fix 2: `distribute_tensor()` function
```python
def distribute_tensor(tensor, layout):
    # Handle TensorLayout objects by extracting the axes tuple
    from keras.src.distribution.distribution_lib import TensorLayout
    if isinstance(backend_layout, TensorLayout):
        # Convert TensorLayout to placements using _layout_to_placements
        placements = _layout_to_placements(backend_layout, tensor, device_mesh)
    elif isinstance(backend_layout, tuple):
        placements = _axis_names_to_placements(backend_layout, device_mesh) if backend_layout else None
    else:
        placements = None
```

#### Fix 3: `distribute_variable()` function
```python
def distribute_variable(tensor, layout=None):
    # CRITICAL FIX: Handle TensorLayout objects properly
    from keras.src.distribution.distribution_lib import TensorLayout
    if isinstance(layout, TensorLayout):
        # Use the axes from TensorLayout
        layout_axes = layout.axes
    else:
        layout_axes = layout
    
    placements = _layout_to_placements(layout_axes, converted_tensor, device_mesh) if layout_axes else None
```

### File: kaggle_opt_hybrid_dp_mp_fixed_v11.py (NEW - V11)
- Fixed regex patterns: `.*feedforward.*` instead of `.*feed_forward.*`
- Added `setup_device_for_rank()` function called BEFORE any CUDA operations
- Added proper distribution setting via `set_distribution()` in redistribution
- Added proper DTensor creation using `torch_distribute_tensor()`
- Added `DTensorParameter` wrapper class for proper DTensor handling
- Added extensive debug logging

## Testing

Run with V12 test:
```bash
torchrun --nproc_per_node=2 kaggle_opt_hybrid_dp_mp_fixed_v11.py
```

Expected results:
- No TypeError about TensorLayout iteration
- No NCCL "Duplicate GPU detected" errors
- Distribution initialization successful
- Weight sharding shows "Sharded: X" where X > 0
- Forward pass completes without errors
- Training step completes successfully

## Key Files Modified
- `keras/src/backend/torch/distribution_lib.py` - Fixed TensorLayout handling (V12)
- `kaggle_opt_hybrid_dp_mp_fixed_v11.py` - New test file with all fixes (V11)

## Debug Commands

Run with NCCL debug:
```bash
NCCL_DEBUG=INFO torchrun --nproc_per_node=2 kaggle_opt_hybrid_dp_mp_fixed_v11.py
```

Run with Keras distribution debug:
```bash
KERAS_DISTRIBUTION_DEBUG=1 torchrun --nproc_per_node=2 kaggle_opt_hybrid_dp_mp_fixed_v11.py
```

