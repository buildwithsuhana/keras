# TODO: Simplify TorchTrainer by consolidating distribution logic

## Goal
Reduce trainer.py by ~300 lines by consolidating sharding/gathering logic into distribution_lib.py

## Steps

### Step 1: Add missing functionality to distribution_lib.py
- [ ] Move `_AllGatherWithGradient` class from trainer.py
- [ ] Move `_all_gather_with_grad` function from trainer.py  
- [ ] Update `dtensor_to_local()` to handle all-gather for sharded tensors with gradient support
- [ ] Verify imports and exports

### Step 2: Simplify trainer.py
- [ ] Delete `_AllGatherWithGradient` class
- [ ] Delete `_all_gather_with_grad` function
- [ ] Delete `_ensure_dtensor_input` method
- [ ] Delete `_convert_to_dtensor_structure` method
- [ ] Delete `_convert_dtensor_output` method
- [ ] Delete `_convert_dtensor_output_structure` method
- [ ] Simplify `_parallelize_if_needed()` to a single call
- [ ] Move `_parallelize_if_needed()` call from fit/evaluate/predict into `_symbolic_build`
- [ ] Update imports to use distribution_lib functions
- [ ] Update `train_step`, `test_step`, `predict_step` to use `tree.map_structure()`
- [ ] Clean up unused imports

### Step 3: Testing
- [ ] Run existing tests to verify no regressions
- [ ] Verify distribution training still works correctly

## Progress
- [x] Plan approved by user
- [x] Implementing Step 1 (distribution_lib.py updates)
  - [x] Moved `_AllGatherWithGradient` class from trainer.py
  - [x] Moved `_all_gather_with_grad` function from trainer.py
  - [x] Updated `dtensor_to_local()` to handle all-gather for sharded tensors
- [x] Implementing Step 2 (trainer.py simplification)
  - [x] Deleted `_AllGatherWithGradient`, `_all_gather_with_grad`
  - [x] Deleted `_ensure_dtensor_input`, `_convert_to_dtensor_structure`
  - [x] Deleted `_convert_dtensor_output`, `_convert_dtensor_output_structure`
  - [x] Simplified `_parallelize_if_needed()` to a single call
  - [x] Moved `_parallelize_if_needed()` call into `_symbolic_build`
  - [x] Updated imports and step functions to use `tree.map_structure()`
- [ ] Testing

