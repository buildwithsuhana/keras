# TODO: Refactor keras/src/backend/torch/trainer.py

## Goal
Move DTensor logic from `trainer.py` to `distribution_lib.py` to keep the trainer focused on the training loop.

## Steps

### Step 1: Add DTensor utilities to distribution_lib.py
- [x] Add `_AllGatherWithGradient` class
- [x] Add `_all_gather_with_grad` function
- [x] Add `prepare_input_for_distribution(x)` function
- [x] Add `prepare_output_for_loss(x)` function

### Step 2: Refactor trainer.py
- [x] Remove `_AllGatherWithGradient` class
- [x] Remove `_all_gather_with_grad` function
- [x] Remove `_ensure_dtensor_input` method
- [x] Remove `_convert_to_dtensor_structure` method
- [x] Remove `_convert_dtensor_output` method
- [x] Remove `_convert_dtensor_output_structure` method
- [x] Add import for distribution_lib utilities
- [x] Update train_step to use distribution_lib utilities
- [x] Update test_step to use distribution_lib utilities
- [x] Update predict_step to use distribution_lib utilities

### Step 3: Verify changes
- [ ] Run tests to ensure refactoring doesn't break functionality

