# TODO: Make parallelize_keras_model() Automatic

## Objective
Eliminate the need for users to manually call `parallelize_keras_model()` by making it automatic during model build/compile.

## Implementation Plan

### Step 1: Add auto-parallelization support to TorchTrainer
- File: `keras/src/backend/torch/trainer.py`
- Override the `_symbolic_build()` method to check for ModelParallel distribution
- Automatically call `_auto_parallelize_model()` when:
  - PyTorch backend is active
  - ModelParallel distribution is set
  - Model has `_torch_layers` attribute
  - Tensor parallel is available

### Step 2: Add helper function to distribution_lib.py
- File: `keras/src/backend/torch/distribution_lib.py`
- Add `_should_auto_parallelize()` function to check if parallelization is needed
- Add `_auto_parallelize_model()` function to handle the parallelization logic
- Add `reset_model_parallelization_state()` function for testing
- Track parallelization state with `_MODEL_PARALLELIZED` global flag

### Step 3: Test the implementation
- Create test script to verify automatic parallelization works
- Verify no duplicate parallelization calls
- Verify manual override still works if needed

## Status
- [x] Step 1: Implement TorchTrainer _symbolic_build() override
- [x] Step 2: Add helper functions to distribution_lib.py
- [ ] Step 3: Test the implementation

