# Plan to Fix Misleading Debug Log Messages in distribution_lib.py

## Issues Identified

### Issue 1: Misleading "fallback" terminology
**Location**: `distribute_variable()` function
**Current message**: 
```
DEBUG | [Rank XX] Manual sharding fallback: shape=..., dtype=...
```
**Problem**: Users seeing "fallback" might think something failed or is suboptimal.

**Fix**: Change to indicate normal operation path:
```
DEBUG | [Rank XX] Using regular Parameter (no sharding needed): shape=..., dtype=...
```

### Issue 2: Confusing "non-floating tensor" messages
**Location**: Multiple places in `distribute_variable()`
**Current messages**:
```
DEBUG | [Rank XX] Non-floating tensor, manual sharding without Parameter: ...
DEBUG | [Rank XX] Non-floating tensor, no mesh: ...
```
**Problem**: These messages make it sound like something went wrong.

**Fix**: Make messages clearer about expected behavior:
```
DEBUG | [Rank XX] Non-floating tensor, returning as-is (no gradient tracking): ...
```

### Issue 3: Misleading "DTensor not available" message
**Location**: `distribute_tensor()` function
**Current message**:
```
DEBUG | DTensor not available, returning tensor as-is
```
**Problem**: This message appears when layout is None, but the code is intentionally not distributing.

**Fix**: 
- Remove the misleading message OR
- Make it conditional on actually being in fallback mode

### Issue 4: Missing context in "returning as-is" messages
**Location**: `distribute_variable()` function
**Current message**:
```
DEBUG | [Rank XX] Non-floating tensor, returning as-is: shape=..., dtype=...
```
**Problem**: Doesn't explain why (expected behavior for non-float tensors).

**Fix**:
```
DEBUG | [Rank XX] Non-floating tensor, returning as-is (no grad tracking needed): shape=..., dtype=...
```

## Changes Summary

### File: `keras/src/backend/torch/distribution_lib.py`

1. **Lines ~585-595**: Change "Manual sharding fallback" to "Using regular Parameter"
2. **Lines ~460-470**: Update "Non-floating tensor, no mesh" message
3. **Lines ~650-660**: Update "Non-floating tensor, manual sharding without Parameter"
4. **Lines ~390-400**: Fix "DTensor not available" message to only show in actual fallback
5. **Lines ~450-460**: Update "Non-floating tensor, returning as-is" with better context

## Expected Behavior After Fix

- Debug logs will clearly indicate when:
  - Normal DTensor distribution is happening
  - Regular Parameters are being created (expected for ModelParallel)
  - Non-floating tensors are handled (expected behavior)
  - Actual fallback paths are taken (when DTensor is unavailable)

- Users will no longer be confused by "fallback" terminology in normal operation paths

