# TODO: Simplify Torch Distribution Lib

## Goal
Reduce `/Users/suhanaaa/keras/keras/src/backend/torch/distribution_lib.py` from ~880 lines to ~300-350 lines while maintaining all functionality.

## Tasks

### 1. Remove Duplicate Re-exports
- [ ] Remove `keras_to_pytorch_path`, `pytorch_to_keras_path`, `convert_path_for_matching` (already in high-level module)

### 2. Simplify distribute_variable Function
- [ ] Consolidate nested conditions into a cleaner flow
- [ ] Reduce from ~200 lines to ~60 lines
- [ ] Maintain ModelParallel and DTensor support

### 3. Remove Unused Functions
- [ ] Remove `_should_auto_parallelize`
- [ ] Remove `_auto_parallelize_model`
- [ ] Remove `_get_keras_layout_map`
- [ ] Remove `_get_torch_module`
- [ ] Remove `_get_tensor_parallel_mesh`

### 4. Strip Debug Logging
- [ ] Remove verbose debug print statements from initialization and distribution functions

### 5. Simplify Initialization
- [ ] Reduce nested conditions in `initialize()`
- [ ] Consolidate environment variable checks

### 6. Consolidate DTensor Functions
- [ ] Merge `_to_dtensor`, `create_replicate_dtensor`, `ensure_dtensor` into single function

### 7. Simplify Mesh Caching
- [ ] Use simpler caching mechanism

### 8. Verify Tests Pass
- [ ] Run distribution_lib_test.py to ensure all tests pass

## Expected Result
- File size: ~300-350 lines (60% reduction)
- All tests passing
- Maintained functionality for DataParallel, ModelParallel, DTensor support

