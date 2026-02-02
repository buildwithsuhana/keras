# Circular Import Fix - Implementation Plan

## Problem
Circular import between:
- `keras/src/distribution/distribution_lib.py` → imports `from keras.src.backend import distribution_lib`
- `keras/src/backend/torch/distribution_lib.py` → imports from `keras.src.distribution.distribution_lib`

## Changes Required

### 1. keras/src/backend/torch/distribution_lib.py
- [ ] Remove top-level imports from `keras.src.distribution.distribution_lib` (lines 167-168)
- [ ] Add lazy import in `distribute_variable()` function
- [ ] Add lazy import in `_to_backend_layout()` function
- [ ] Keep the import of path_utils at the end (it doesn't cause circular imports)

### 2. Testing
- [ ] Run the import test to verify fix works

