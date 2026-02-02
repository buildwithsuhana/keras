# Circular Import Fix - TODO List

## Problem
Circular import between:
- `keras/src/backend/torch/__init__.py` → imports `distribution_lib`
- `keras/src/backend/torch/distribution_lib.py` → imports from `keras.src.distribution.distribution_lib`
- `keras/src/distribution/distribution_lib.py` → imports `from keras.src.backend import distribution_lib`

## Solution
Break the cycle by using lazy imports in `keras/src/backend/torch/distribution_lib.py`

## Changes

### 1. keras/src/backend/torch/__init__.py
- [ ] Remove line 46: `from keras.src.backend.torch import distribution_lib`

### 2. keras/src/backend/torch/distribution_lib.py
- [ ] Remove lines 340-349 (imports from `keras.src.distribution.distribution_lib`)
- [ ] Add lazy imports in `distribute_variable()` function
- [ ] Add lazy imports in `_to_backend_layout()` function  
- [ ] Add lazy imports in `DataParallel.__init__()` method
- [ ] Add lazy imports in `ModelParallel.__init__()` method
- [ ] Add lazy imports in `Distribution.scope()` method

