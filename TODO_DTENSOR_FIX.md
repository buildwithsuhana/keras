# DTensor Fix Implementation Plan

## Task
Fix the `RuntimeError: aten.sub.Tensor: got mixed torch.Tensor and DTensor` error when using Keras ModelParallel with Torch backend.

## Root Cause
When `torch.compile`/`dynamo` is enabled, optimizer internal variables (learning_rate, beta_1, beta_2, epsilon, etc.) interact with DTensor model weights during `torch._foreach_*` operations. If optimizer variables are still regular torch.Tensor and model weights are DTensor, PyTorch raises the mixed tensor error.

## Solution
Convert ALL scalar values used in torch._foreach_* operations to native Python floats. This ensures that foreach operations receive compatible native scalars rather than mixing DTensor and regular tensors.

## Implementation Steps

### Step 1: Update torch_adam.py
- Convert `lr` to native scalar (already done)
- Convert `alpha` to native scalar (already done)
- Convert `self.beta_1`, `self.beta_2` to native scalars
- Convert `self.epsilon` to native scalar

### Step 2: Update torch_adagrad.py
- Convert `lr` to native scalar (already done)
- Convert `self.epsilon` to native scalar

### Step 3: Update torch_sgd.py
- Convert `learning_rate` to native scalar (already done)
- Convert `self.momentum` to native scalar

### Step 4: Update torch_lion.py
- Convert `lr` to native scalar (already done)
- Convert `self.beta_1`, `self.beta_2` to native scalars
- Convert `self.epsilon` to native scalar

### Step 5: Update torch_adamax.py
- Convert `lr` to native scalar (already done)
- Convert `den_scalar` to native scalar (already done)
- Convert `self.beta_2` to native scalar
- Convert `self.epsilon` to native scalar

### Step 6: Update torch_adadelta.py
- Convert `lr` to native scalar (already done)
- Convert `self.epsilon` to native scalar
- Convert `rho` to native scalar (if used in foreach)

### Step 7: Update torch_rmsprop.py
- Convert `lr` to native scalar (already done)
- Convert `self.epsilon` to native scalar
- Convert `rho` to native scalar (if used in foreach)

### Step 8: Update torch_nadam.py
- Convert `lr` to native scalar (already done)
- Convert `u_t`, `u_t_1` to native scalars (already done)
- Convert `self.beta_1`, `self.beta_2` to native scalars
- Convert `self.epsilon` to native scalar

### Step 9: Test the fix
Run the training script to verify the fix works.

## Files to Modify
- `/Users/suhanaaa/keras/keras/src/backend/torch/optimizers/torch_adam.py`
- `/Users/suhanaaa/keras/keras/src/backend/torch/optimizers/torch_adagrad.py`
- `/Users/suhanaaa/keras/keras/src/backend/torch/optimizers/torch_sgd.py`
- `/Users/suhanaaa/keras/keras/src/backend/torch/optimizers/torch_lion.py`
- `/Users/suhanaaa/keras/keras/src/backend/torch/optimizers/torch_adamax.py`
- `/Users/suhanaaa/keras/keras/src/backend/torch/optimizers/torch_adadelta.py`
- `/Users/suhanaaa/keras/keras/src/backend/torch/optimizers/torch_rmsprop.py`
- `/Users/suhanaaa/keras/keras/src/backend/torch/optimizers/torch_nadam.py`

## Progress
- [ ] Update torch_adam.py
- [ ] Update torch_adagrad.py
- [ ] Update torch_sgd.py
- [ ] Update torch_lion.py
- [ ] Update torch_adamax.py
- [ ] Update torch_adadelta.py
- [ ] Update torch_rmsprop.py
- [ ] Update torch_nadam.py
- [ ] Test the fix

