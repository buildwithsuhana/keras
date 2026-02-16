# TODO: Fix Mixed Tensor Error in PyTorch Distributed Optimizers

## Problem
The error "aten._foreach_add_.List: got mixed torch.Tensor and DTensor" occurs during training with DataParallel because:
1. Model weights and optimizer states are converted to DTensors
2. But gradients remain as regular torch.Tensors
3. When torch._foreach_* operations are called with mixed types, they fail

## Files to Check/Update
1. keras/src/backend/torch/optimizers/torch_adam.py - Already has fix, verify
2. keras/src/backend/torch/optimizers/torch_sgd.py - Already has fix, verify
3. keras/src/backend/torch/optimizers/torch_rmsprop.py - FIXED: conversion moved to BEFORE usage
4. keras/src/backend/torch/optimizers/torch_adagrad.py - Already has fix, verify
5. keras/src/backend/torch/optimizers/torch_adadelta.py - Already has fix, verify
6. keras/src/backend/torch/optimizers/torch_lion.py - Already has fix, verify
7. keras/src/backend/torch/optimizers/torch_nadam.py - Already has fix, verify
8. keras/src/backend/torch/optimizers/torch_adamax.py - Already has fix, verify

## Fixes Applied
- torch_rmsprop.py: Moved `_convert_grads_to_dtensor()` call to BEFORE the first `torch._foreach_*` operation that uses gradients

## Key Fix Required
The `_convert_grads_to_dtensor()` call MUST happen BEFORE any `torch._foreach_*` operations that:
1. Take both gradients AND optimizer states as arguments
2. Or operate on optimizer states that were previously updated with gradients

## Status
- [x] Review each optimizer file
- [x] Fix torch_rmsprop.py - move conversion to BEFORE first usage
- [ ] Test the fix

