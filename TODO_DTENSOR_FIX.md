# DTensor Fix Implementation Plan

## Task
Fix the `RuntimeError: aten.index.Tensor: got mixed torch.Tensor and DTensor` error when using Keras ModelParallel with Torch backend.

## Root Cause
When using DTensor model weights with a torch DataLoader, PyTorch's internal `tree_map` operations in `__getitems__` fail because they mix regular `torch.Tensor` and `DTensor`. This happens because:
1. The model weights are converted to DTensor for model parallelism
2. But the data from DataLoader remains as regular torch.Tensor
3. PyTorch's internal operations (like `tree_map` in DataLoader) can't handle the mix

## Solution
Convert input data to DTensors when distribution is active, before the DataLoader processes it.

## Implementation Steps

### Step 1: Update distribution_lib.py ✅
- Add `_get_dtensor_mesh_and_placements()` helper function
- Add `_convert_to_dtensor()` helper function  
- Add `_convert_batch_to_dtensor()` helper function

### Step 2: Update torch_data_loader_adapter.py ✅
- Modify `get_torch_dataloader()` to return a wrapper that handles DTensor conversion
- Add `_DTensorAwareDataLoader` class
- Add `_DTensorAwareDataset` class

### Step 3: Update epoch_iterator.py ✅
- Ensure `_distribute_data()` properly converts numpy arrays to tensors before distribution
- Handle the case where distribution is active but data is numpy

### Step 4: Test the fix
Run the training script to verify the fix works.

## Files Modified
- `/Users/suhanaaa/keras/keras/src/backend/torch/distribution_lib.py`
- `/Users/suhanaaa/keras/keras/src/trainers/data_adapters/torch_data_loader_adapter.py`
- `/Users/suhanaaa/keras/keras/src/trainers/epoch_iterator.py`

## Progress
- [x] Update distribution_lib.py
- [x] Update torch_data_loader_adapter.py
- [x] Update epoch_iterator.py
- [ ] Test the fix

