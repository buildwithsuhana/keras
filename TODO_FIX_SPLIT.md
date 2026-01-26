# Fix Plan for Distributed Training on 2 GPUs

## Issues Identified

1. **TypeError: split_with_sizes()**: `torch.split()` receives `numpy.int64` instead of Python int
2. **Shape mismatch (32x128 and 32x1024)**: Weights sharded along 'model' axis but inputs not properly distributed

## Root Cause Analysis

The error occurs in two places:
1. `numpy.py` `split()` function - chunk_sizes may be numpy.int64
2. `distribution_lib.py` `_shard_tensor_native()` - dim_size may be numpy type

## Fixes to Implement

### Fix 1: numpy.py - Ensure chunk_sizes are native Python ints
**File**: `keras/src/backend/torch/numpy.py`
**Function**: `split()`
**Change**: Convert chunk_sizes to tuple of Python ints before passing to torch.split

### Fix 2: distribution_lib.py - Ensure dim_size is native Python int
**File**: `keras/src/backend/torch/distribution_lib.py`
**Function**: `_shard_tensor_native()`
**Change**: Convert dim_size and chunk_size to native Python ints

## Files Modified
1. `keras/src/backend/torch/numpy.py`
2. `keras/src/backend/torch/distribution_lib.py`

## Steps
- [x] Fix split() in numpy.py to ensure chunk_sizes are Python ints
- [x] Fix _shard_tensor_native() in distribution_lib.py to ensure dim_size is Python int
- [ ] Test the fixes

