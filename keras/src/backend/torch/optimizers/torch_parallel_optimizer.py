import os
import torch

from keras.src.optimizers.base_optimizer import BaseOptimizer
from keras.src.utils import torch_utils


def _convert_grads_to_dtensor(grads, variables):
    """Convert regular torch.Tensor gradients to DTensor to match optimizer states.
    
    When training with DTensor (PyTorch distributed), the optimizer states
    (momentum, velocity) are converted to DTensors via distribute_variable().
    However, gradients remain as regular torch.Tensors. This causes a mismatch
    when torch._foreach_* operations are called, as they cannot handle mixed
    types.
    
    This function converts gradients to DTensors with the same placements as
    the corresponding optimizer state variables.
    """
    # Skip if not in distributed context
    if not torch.distributed.is_initialized():
        return grads
    
    # Import here to avoid circular imports
    from torch.distributed._tensor import DTensor, Replicate
    from keras.src.backend.torch.distribution_lib import _get_default_device_mesh
    
    device_mesh = _get_default_device_mesh()
    if device_mesh is None:
        return grads
    
    # Check if any variable is a DTensor
    has_dtensor = False
    for v in variables:
        value = getattr(v, 'value', v)
        if isinstance(value, DTensor):
            has_dtensor = True
            break
    
    if not has_dtensor:
        return grads
    
    # Convert grads to DTensors with the same placements as variables
    converted_grads = []
    for grad, variable in zip(grads, variables):
        if grad is None:
            converted_grads.append(None)
            continue
            
        value = getattr(variable, 'value', variable)
        if isinstance(value, DTensor):
            # Use the same placements as the optimizer state variable
            placements = value.placements
            dtensor = DTensor.from_local(grad, device_mesh, placements)
            converted_grads.append(dtensor)
        else:
            # For non-DTensor variables, replicate the gradient
            converted_grads.append(grad)
    
    return converted_grads


class TorchParallelOptimizer(BaseOptimizer):
    @torch_utils.no_grad
    def _backend_update_step(self, grads, trainable_variables, learning_rate):
        self._parallel_update_step(
            grads,
            trainable_variables,
            learning_rate,
        )

    @torch_utils.no_grad
    def _backend_reset_gradient_accumulators(self):
        acc_list = [
            v.value for v in self._accumulated_gradients if v is not None
        ]
        torch._foreach_mul_(acc_list, 0.0)

    @torch_utils.no_grad
    def _backend_increment_gradient_accumulators(self, grads, acc_grads):
        acc_list = [v.value for v in acc_grads]
        torch._foreach_add_(acc_list, grads, alpha=1.0)

