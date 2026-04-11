# TODO: Fix performance issue in torch/core.py convert_to_tensor

## Steps:
1. [x] Add `from torch.distributed.tensor import DTensor` to the imports in `keras/src/backend/torch/core.py`.
2. [x] Update the if-condition in `convert_to_tensor` to include `and not isinstance(x, DTensor)` before creating TensorLayout.
3. [ ] Verify the changes and complete the task.

