# TODO: Fix OPT Model Parallelism Sharding

## Problem Summary
The OPT model with hybrid Data Parallel + Model Parallel has issues:
1. Sharding not properly applied - weights show "Sharded: 0"
2. Runtime error during forward pass: "The size of tensor a (6) must match the size of tensor b (12)"
3. Pattern mismatch: `.*feed_forward.*` doesn't match `feedforward` (underscore issue)
4. NCCL "Duplicate GPU detected" - both ranks using same GPU
5. Missing module: `ModuleNotFoundError: No module named 'torch.distributed.tensor.api'`
6. `distribution()` returns `None` during forward pass

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

## Changes Made

### File: keras/src/backend/torch/distribution_lib.py
- Fixed `_layout_to_placements()` to shard on correct tensor dimension (V10)
- Improved `initialize()` to set CUDA device BEFORE distributed init (V11)
- Added proper device assignment: `gpu_id = local_rank % num_gpus`

### File: kaggle_opt_hybrid_dp_mp_fixed_v11.py (NEW)
- Fixed regex patterns: `.*feedforward.*` instead of `.*feed_forward.*`
- Added `setup_device_for_rank()` function called BEFORE any CUDA operations
- Added proper distribution setting via `set_distribution()` in redistribution
- Added proper DTensor creation using `torch_distribute_tensor()`
- Added `DTensorParameter` wrapper class for proper DTensor handling
- Added extensive debug logging

## Testing

Run with V11 test:
```bash
torchrun --nproc_per_node=2 kaggle_opt_hybrid_dp_mp_fixed_v11.py
```

Expected results:
- No NCCL "Duplicate GPU detected" errors
- Distribution initialization successful
- Weight sharding shows "Sharded: X" where X > 0
- Forward pass completes without errors
- Training step completes successfully

## Key Files Modified
- `keras/src/backend/torch/distribution_lib.py` - Fixed initialization and placement conversion
- `kaggle_opt_hybrid_dp_mp_fixed_v11.py` - New test file with all fixes

## Debug Commands

Run with NCCL debug:
```bash
NCCL_DEBUG=INFO torchrun --nproc_per_node=2 kaggle_opt_hybrid_dp_mp_fixed_v11.py
```

Run with Keras distribution debug:
```bash
KERAS_DISTRIBUTION_DEBUG=1 torchrun --nproc_per_node=2 kaggle_opt_hybrid_dp_mp_fixed_v11.py
```

