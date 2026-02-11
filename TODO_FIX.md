# TODO: Fix OPT Model Parallelism Sharding

## Problem Summary
The OPT model with hybrid Data Parallel + Model Parallel has issues:
1. Sharding not properly applied - weights show "Sharded: 0"
2. Runtime error during forward pass: "The size of tensor a (6) must match the size of tensor b (12)"
3. **Pattern mismatch: `re.match()` only matches at BEGINNING of string** - FIXED in V13
4. NCCL "Duplicate GPU detected" - both ranks using same GPU
5. Missing module: `ModuleNotFoundError: No module named 'torch.distributed.tensor.api'`
6. `distribution()` returns `None` during forward pass
7. **TypeError: 'TensorLayout' object is not iterable** - Fixed in V12
8. **Mixed tensor types error** - Fixed in V12 with input DTensor conversion

## Root Causes Identified

### 1. Pattern Mismatch - WRONG REGEX FUNCTION (FIXED in V13)
- **V12 ISSUE**: Code used `re.match()` which only matches at the **beginning** of the string
- Actual weight name: `"transformer_layer_0/feedforward_intermediate_dense/kernel"`
- Pattern: `".*feedforward.*intermediate.*dense.*kernel"`
- `re.match()` fails because the string doesn't START with `.*feedforward...`
- **V13 FIX**: Changed to `re.search()` which matches **anywhere** in the string
- `re.search()` finds `feedforward` anywhere in the path

### 2. Pattern Mismatch - Underscore Issue (FIXED in V11)
- Pattern: `".*feed_forward.*kernel"` 
- Actual weight name: `"transformer_layer_0/feedforward_intermediate_dense/kernel"`
- Issue: underscore vs no underscore
- Fix: Changed pattern to `".*feedforward.*kernel"` (no underscore)

### 3. Wrong Dimension for Sharding (FIXED in V10)
- In `_layout_to_placements()`:
  - Layout `(None, 'model')` should shard on tensor dimension 1
  - Original code returned `[Shard(0)]` (mesh dimension)
  - Fixed to return `[Shard(1)]` (tensor dimension where 'model' appears)

### 4. NCCL "Duplicate GPU detected" Error (FIXED in V11)
- Both ranks trying to use same GPU because proper CUDA device assignment wasn't working
- Fix: Set CUDA device BEFORE any torch.cuda calls or distributed init
- Each rank uses unique GPU: `gpu_id = local_rank % num_gpus`

### 5. Missing `torch.distributed.tensor.api` Module (FIXED)
- Wrong import path used in some places
- Correct import: `from torch.distributed._tensor.api import distribute_tensor as torch_distribute_tensor`

### 6. `distribution()` returns `None` (FIXED in V11)
- Distribution context not properly set during forward pass
- Fix: Ensure `set_distribution()` is called in the test script before redistribution

### 7. TypeError: 'TensorLayout' object is not iterable (FIXED in V12)
- The `_layout_to_placements()` function tried to iterate directly over `TensorLayout` objects
- Fix: Check if layout is a `TensorLayout` instance and extract `.axes` before iterating
- Similar fix applied to `distribute_tensor()` and `distribute_variable()` functions

### 8. Mixed tensor types error (FIXED in V12)
- Regular `torch.Tensor` inputs passed to model with `DTensor` weights
- Fix: Convert inputs to DTensors using `prepare_input_for_distribution()` function

## Changes Made in V13

### File: kaggle_opt_hybrid_dp_mp_fixed_v13.py

#### Fix: Use re.search() instead of re.match() for pattern matching
```python
# BEFORE (V12 - WRONG):
if re.match(pattern, v.path):
    target_layout = layout
    matched_pattern = pattern
    ...

# AFTER (V13 - CORRECT):
if re.search(pattern, v.path):
    target_layout = layout
    matched_pattern = pattern
    ...
```

**Why this fix is critical:**
- `re.match()` only matches at the **beginning** of the string
- Variable paths: `"transformer_layer_0/feedforward_intermediate_dense/kernel"`
- Patterns: `".*feedforward.*intermediate.*dense.*kernel"`
- `re.match()` fails because the string starts with "transformer_layer_0", not ".*feedforward"
- `re.search()` correctly finds "feedforward" anywhere in the path

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

### File: kaggle_opt_hybrid_dp_mp_fixed_v12.py (NEW - V12)
- Fixed regex patterns: `.*feedforward.*` instead of `.*feed_forward.*`
- Added `setup_device_for_rank()` function called BEFORE any CUDA operations
- Added proper distribution setting via `set_distribution()` in redistribution
- Added proper DTensor creation using `torch_distribute_tensor()`
- Added `DTensorParameter` wrapper class for proper DTensor handling
- Added extensive debug logging
- Added `prepare_input_for_distribution()` function to convert inputs to DTensors

## Testing

Run with V13 test:
```bash
torchrun --nproc_per_node=2 kaggle_opt_hybrid_dp_mp_fixed_v13.py
```

Expected results with V13:
- Pattern matching now works: `re.search()` finds patterns anywhere in variable path
- More variables should match: Expected 100+ matches instead of 2
- No more "NO MATCH" debug messages for weights like:
  - `transformer_layer_0/feedforward_intermediate_dense/kernel`
  - `transformer_layer_0/self_attention/query/kernel`
- Weight redistribution successful: Sharded weights show "Sharded: X" where X > 0
- Forward pass completes without "mixed torch.Tensor and DTensor" errors

## Key V12 Additions

### 1. Input DTensor Conversion (CRITICAL)
```python
def prepare_input_for_distribution(x, device_mesh=None):
    """Convert input tensor to DTensor if distribution is enabled.
    
    This is crucial for avoiding "mixed torch.Tensor and DTensor" errors
    during distributed forward passes.
    """
    if isinstance(x, np.ndarray):
        x_tensor = torch.from_numpy(x)
    elif isinstance(x, torch.Tensor):
        x_tensor = x
    
    if torch.cuda.is_available():
        x_tensor = x_tensor.cuda()
    
    # Convert to DTensor with Replicate placement
    dtensor = torch_distribute_tensor(x_tensor, device_mesh, [Replicate()])
    return dtensor
```

### 2. Fixed Regex Patterns
```python
# Use patterns that properly match with underscores
layout_map[".*self_attention.*query.*kernel"] = (None, "model")
layout_map[".*self_attention.*key.*kernel"] = (None, "model")
layout_map[".*self_attention.*value.*kernel"] = (None, "model")
layout_map[".*self_attention.*output.*kernel"] = (None, "model")

layout_map[".*feedforward.*intermediate.*dense.*kernel"] = (None, "model")
layout_map[".*feedforward.*output.*dense.*kernel"] = (None, "model")
```

## Debug Commands

Run with NCCL debug:
```bash
NCCL_DEBUG=INFO torchrun --nproc_per_node=2 kaggle_opt_hybrid_dp_mp_fixed_v13.py
```

Run with Keras distribution debug:
```bash
KERAS_DISTRIBUTION_DEBUG=1 torchrun --nproc_per_node=2 kaggle_opt_hybrid_dp_mp_fixed_v13.py
```

## Expected Behavior After V13 Fix

### Pattern Matching Debug Output (AFTER FIX)
```
[Rank 0] LayoutMap patterns:
  Pattern: '.*feedforward.*intermediate.*dense.*kernel' -> layout (None, 'model')
  ...

[Rank 0] Using DeviceMesh: shape=torch.Size([2])

  DEBUG: REGEX MATCH - Pattern '.*feedforward.*intermediate.*dense.*kernel' matches 'transformer_layer_0/feedforward_intermediate_dense/kernel' -> layout (None, 'model')
  DEBUG: REGEX MATCH - Pattern '.*self_attention.*query.*kernel' matches 'transformer_layer_0/self_attention/query/kernel' -> layout (None, 'model')
  DEBUG: REGEX MATCH - Pattern '.*feedforward.*output.*dense.*kernel' matches 'transformer_layer_0/feedforward_output_dense/kernel' -> layout (None, 'model')
  ...
  
[Rank 0] Summary:
  Total patterns: 20
  Variables matched: 142
  Variables redistributed: 142
```

## Key Files Modified
- `kaggle_opt_hybrid_dp_mp_fixed_v13.py` - New test file with V13 fix (re.search instead of re.match)
- `keras/src/backend/torch/distribution_lib.py` - Fixed TensorLayout handling (V12)
- `kaggle_opt_hybrid_dp_mp_fixed_v12.py` - Test file with V12 fixes

## Summary of All Fixes

| Fix | Version | Description |
|-----|---------|-------------|
| TensorLayout iteration | V12 | Handle TensorLayout objects properly in _layout_to_placements |
| Mixed tensor inputs | V12 | Convert inputs to DTensors before forward pass |
| Underscore in patterns | V11/V12 | Use `.*feedforward.*` instead of `.*feed_forward.*` |
| NCCL Duplicate GPU | V11 | Set CUDA device before distributed init |
| distribution() returns None | V11 | Call set_distribution() before redistribution |
| Wrong sharding dimension | V10 | Return [Shard(1)] for 'model' axis, not [Shard(0)] |
| **re.match() vs re.search()** | **V13** | **Use re.search() to match patterns anywhere in path** |
| **Pattern conflicts** | **V14** | **Remove generic `.*layer_norm.*` patterns, use specific ones** |

## Changes Made in V14

### File: kaggle_opt_hybrid_dp_mp_fixed_v14.py

#### Fix 1: Remove conflicting generic layer_norm patterns
```python
# V14 FIX: Use SPECIFIC layer norm patterns only, no generic ones
# These match paths like: "transformer_layer_0/self_attention_layer_norm/gamma"
layout_map[".*self_attention_layer_norm.*gamma"] = ()
layout_map[".*self_attention_layer_norm.*beta"] = ()
layout_map[".*feedforward_layer_norm.*gamma"] = ()
layout_map[".*feedforward_layer_norm.*beta"] = ()

# V14 FIX: Remove generic `.*layer_norm.*` patterns - they cause conflicts!
# DO NOT include: layout_map[".*layer_norm.*gamma"] = ()  # CONFLICTS!
```

**Why this fix is critical:**
- Variable path: `"transformer_layer_0/self_attention_layer_norm/gamma"`
- Pattern `".*self_attention.*layer_norm.*gamma"` matches
- Pattern `".*layer_norm.*gamma"` ALSO matches
- Keras throws: "Path matches multiple layout specification keys"

#### Fix 2: Skip DataLoader in training, use full batch
```python
# V14 FIX: Pass data directly, not through DataLoader
# This avoids the "mixed torch.Tensor and DTensor" error
history = model.fit(
    train_x_dtensor, train_y_dtensor,
    epochs=1,
    batch_size=16,  # Use full batch to avoid DataLoader
    verbose=1,
    shuffle=False  # Disable shuffling to avoid DataLoader
)
```

**Why this fix is critical:**
- Keras DataLoader internally creates regular torch.Tensor
- DTensor weights + torch.Tensor from DataLoader = "mixed tensor" error
- Using full batch (batch_size=16) with data directly bypasses DataLoader

