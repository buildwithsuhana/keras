# TorchTrainer DTensor Support Simplification

## Goal
Simplify the DTensor support implementation in `keras/src/backend/torch/trainer.py` to reduce code complexity and lines of code.

## Changes Made

### 1. Merge Conversion Functions
- [x] Keep `_AllGatherWithGradient` and `_all_gather_with_grad` (core functionality)
- [x] Create unified `_convert_dtensor_structure` function for nested structure handling
- [x] Merge `_ensure_dtensor_input` and `_convert_dtensor_output` into `_handle_dtensor`

### 2. Simplify Step Functions
- [x] Reduce code duplication in train_step, test_step, predict_step
- [x] Created `_unpack_data` helper for consistent DTensor handling

### 3. Simplify _parallelize_if_needed
- [x] Reduce nested try-except blocks
- [x] Streamline the parallelization logic

### 4. Code Reduction
- [x] From ~600 lines to ~400 lines (33% reduction)

## Summary of Simplified Code

### Before (Original):
- 8 functions for DTensor handling
- Complex nested logic
- ~600 lines

### After (Simplified):
- 4 key functions for DTensor handling:
  1. `_get_dtensor_context()` - Get distribution context once
  2. `_convert_dtensor_structure()` - Unified conversion for nested structures
  3. `_handle_dtensor()` - Single entry point for input/output conversion
  4. `_AllGatherWithGradient` - Custom autograd (kept as-is)
- Clean, linear logic flow
- ~400 lines (33% reduction)

## Key Simplifications

1. **Single context function**: `_get_dtensor_context()` consolidates all imports and checks
2. **Unified conversion**: `_convert_dtensor_structure` handles both input (tensor→DTensor) and output (DTensor→local with all-gather)
3. **Single entry point**: `_handle_dtensor()` with `is_output` flag
4. **Consistent step functions**: All use `_unpack_data()` helper

## Files Modified
- `keras/src/backend/torch/trainer.py`

