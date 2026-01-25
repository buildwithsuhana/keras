# DTensor Fix Implementation Plan

## Task
Fix the `RuntimeError: aten.sub.Tensor: got mixed torch.Tensor and DTensor` error when using Keras ModelParallel with Torch backend.

## Root Cause
When `torch.compile`/`dynamo` is enabled, optimizer internal variables (learning_rate, iteration, etc.) interact with DTensor model weights during `torch._foreach_*` operations. If optimizer variables are still regular torch.Tensor and model weights are DTensor, PyTorch raises the mixed tensor error.

## Solution
Convert all scalar tensors used in torch._foreach_* operations to native Python floats. This ensures that foreach operations receive compatible native scalars rather than mixing DTensor and regular tensors.

## Implementation Steps

### Step 1: Add helper function to torch_parallel_optimizer.py
Add `_to_native_scalar()` helper that extracts native Python values from DTensor scalars.

### Step 2: Update torch_parallel_optimizer.py
Add `_to_native_scalar()` helper and modify `_backend_update_step` to convert scalar values.

### Step 3: Update all torch optimizer parallel update methods
- torch_adagrad.py - convert lr to native scalar
- torch_nadam.py - convert u_t, u_t_1 to native scalars
- torch_adam.py - convert lr, betas to native scalars
- torch_adamax.py - convert lr to native scalars  
- torch_adadelta.py - convert lr, rho to native scalars
- torch_rmsprop.py - convert lr to native scalars
- torch_sgd.py - convert lr to native scalars
- torch_lion.py - convert lr to native scalars
- torch_adamw.py - convert lr to native scalars (inherits from Adam)

### Step 4: Update torch_optimizer.py
Update `_apply_weight_decay` to convert learning_rate to native scalar.

### Step 5: Test the fix
Run the unit tests to verify the fix works.

## Files Modified
- `/Users/suhanaaa/keras/keras/src/backend/torch/optimizers/torch_parallel_optimizer.py` - Added `_to_native_scalar()` helper
- `/Users/suhanaaa/keras/keras/src/backend/torch/optimizers/torch_optimizer.py` - Updated `_apply_weight_decay`
- `/Users/suhanaaa/keras/keras/src/backend/torch/optimizers/torch_adagrad.py` - Simplified scalar conversion
- `/Users/suhanaaa/keras/keras/src/backend/torch/optimizers/torch_nadam.py` - Simplified scalar conversion
- `/Users/suhanaaa/keras/keras/src/backend/torch/optimizers/torch_adam.py` - Simplified scalar conversion
- `/Users/suhanaaa/keras/keras/src/backend/torch/optimizers/torch_adamax.py` - Simplified scalar conversion
- `/Users/suhanaaa/keras/keras/src/backend/torch/optimizers/torch_adadelta.py` - Simplified scalar conversion
- `/Users/suhanaaa/keras/keras/src/backend/torch/optimizers/torch_rmsprop.py` - Simplified scalar conversion
- `/Users/suhanaaa/keras/keras/src/backend/torch/optimizers/torch_sgd.py` - Simplified scalar conversion
- `/Users/suhanaaa/keras/keras/src/backend/torch/optimizers/torch_lion.py` - Simplified scalar conversion
- `/Users/suhanaaa/keras/keras/src/backend/torch/optimizers/torch_adamw.py` - Inherits from Adam, no changes needed

## Progress
- [x] Add helper function to torch_parallel_optimizer.py
- [x] Update torch_parallel_optimizer.py
- [x] Update torch_adagrad.py
- [x] Update torch_nadam.py
- [x] Update torch_adam.py
- [x] Update torch_adamax.py
- [x] Update torch_adadelta.py
- [x] Update torch_rmsprop.py
- [x] Update torch_sgd.py
- [x] Update torch_lion.py
- [x] Update torch_adamw.py (inherits from Adam)
- [x] Update torch_optimizer.py
- [ ] Test the fix

