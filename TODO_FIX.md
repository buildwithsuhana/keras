# TODO: Fix OPT Model Parallelism Sharding

## Problem Summary
The OPT model with hybrid Data Parallel + Model Parallel has issues:
1. Sharding not properly applied - weights show "Sharded: 0"
2. Runtime error during forward pass: "The size of tensor a (6) must match the size of tensor b (12)"
3. Pattern mismatch: `.*feed_forward.*` doesn't match `feedforward` (underscore issue)

## Root Causes Identified

### 1. Pattern Mismatch (FIXED)
- Pattern: `".*feed_forward.*kernel"` 
- Actual weight name: `"transformer_layer_0/feedforward_intermediate_dense/kernel"`
- Issue: underscore vs no underscore
- Fix: Changed pattern to `".*feedforward.*kernel"` (no underscore)

### 2. Wrong Dimension for Sharding (FIXED)
- In `_layout_to_placements()`:
  - Layout `(None, 'model')` should shard on tensor dimension 1
  - Original code returned `[Shard(0)]` (mesh dimension)
  - Fixed to return `[Shard(1)]` (tensor dimension where 'model' appears)

### 3. DTensor Not Properly Created
- Manual slicing modified `_value` attribute but didn't create proper DTensors
- The attention layer expects full tensor operations but receives sharded ones
- Need to use `torch_distribute_tensor()` to create proper DTensors

## Changes Made

### File: keras/src/backend/torch/distribution_lib.py
- Fixed `_layout_to_placements()` to shard on correct tensor dimension
- Layout `(None, 'model')` now returns `[Shard(1)]` instead of `[Shard(0)]`

### File: kaggle_opt_hybrid_dp_mp_fixed_v10.py (NEW)
- Fixed regex patterns: `.*feedforward.*` instead of `.*feed_forward.*`
- Added proper DTensor creation using `torch_distribute_tensor()`
- Added `DTensorParameter` wrapper class for proper DTensor handling
- Added extensive debug logging

## Next Steps

1. Test the fixed version:
   ```bash
   torchrun --nproc_per_node=2 kaggle_opt_hybrid_dp_mp_fixed_v10.py
   ```

2. Verify:
   - Sharding summary shows "Sharded: X" where X > 0
   - Forward pass completes without errors
   - Training step completes successfully

3. If issues persist:
   - Check if the DTensorParameter wrapper is correctly handling the DTensor
   - Verify the attention layer understands DTensor sharding
   - Consider using PyTorch's `parallelize_module()` for automatic handling

## Key Files Modified
- `keras/src/backend/torch/distribution_lib.py` - Fixed placement conversion
- `kaggle_opt_hybrid_dp_mp_fixed_v10.py` - New test file with fixes

## Alternative Solutions to Consider

1. **Use PyTorch parallelize_module**: Instead of manual sharding, use PyTorch's `parallelize_module()` with a layout plan

2. **Implement Custom Attention Layer**: Create an attention layer that understands DTensor sharding and does proper collective operations

3. **All-Gather Before Attention**: Gather the sharded weights before the attention computation, then shard again after

## Debug Commands

Run with NCCL debug:
```bash
NCCL_DEBUG=INFO torchrun --nproc_per_node=2 kaggle_opt_hybrid_dp_mp_fixed_v10.py
```

Run with Keras distribution debug:
```bash
KERAS_DISTRIBUTION_DEBUG=1 torchrun --nproc_per_node=2 kaggle_opt_hybrid_dp_mp_fixed_v10.py
```

