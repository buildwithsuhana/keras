# TODO: Fix Manual Sharding for ModelParallel

## Status: IN PROGRESS

### Analysis Complete
- `distribute_variable()` in `torch/distribution_lib.py` already has logic to detect ModelParallel
- When ModelParallel is active, it creates regular `torch.nn.Parameter` instead of DTensor
- The `parallelize_module` should handle the actual DTensor conversion

### Issue Identified
The `_parallelize_if_needed()` method in `TorchTrainer` should be called at the right time to parallelize the model before training.

### Changes Made
1. `core.py` - Enhanced `_initialize_layout()` with better debug logging

### Remaining Work
1. Test the current implementation
2. Verify that `_parallelize_if_needed()` is called at the right time
3. Ensure parallelize_module properly converts Parameters to DTensors

## Testing Commands

```bash
# Single process test (with CPU)
KERAS_BACKEND=torch KERAS_DISTRIBUTION_DEBUG=1 python kaggle_distributed_test.py

# Multi-process test
torchrun --nproc_per_node=2 kaggle_distributed_test.py
```

## Expected Results
- DataParallel: Variables replicated (not sharded)
- ModelParallel: Regular Parameters created, parallelize_module handles sharding
- No mixed tensor errors
- Training completes successfully

