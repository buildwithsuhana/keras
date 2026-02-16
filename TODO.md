# TODO: Fix ModelParallel Multi-Process Input Sharding Issue

## Problem
When running ModelParallel with multi-process (e.g., 2 GPUs), the training fails with:
```
RuntimeError: a and b must have same reduction dim, but got [32, 256] X [128, 512].
```

This happens because:
1. The kernel weights are correctly sharded as DTensors
2. The inputs are correctly kept as local tensors (not DTensors)
3. But the matmul operation is failing due to shape mismatch

## Root Cause
The issue is in the `_layout_to_placements` function. When the 2D mesh (shape=(1,2)) falls back to 1D mesh in multi-process mode, the layout conversion doesn't properly handle the case where:
- Input is a local tensor [batch, input_dim]
- Kernel is sharded DTensor [input_dim, units/shard]

The DTensor sharding propagation is computing wrong intermediate shapes during tracing.

## Solution Plan

1. **Fix `_layout_to_placements` in `distribution_lib.py`**:
   - Ensure correct placement mapping when falling back from 2D to 1D mesh
   - The placement should shard on tensor dimension 1 (output dimension) using the single mesh dimension

2. **Test the fix**:
   - Run the distributed test with ModelParallel

## Files to Modify
- `keras/src/backend/torch/distribution_lib.py`

