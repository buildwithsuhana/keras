import os
import torch

from keras.src.optimizers.base_optimizer import BaseOptimizer
from keras.src.utils import torch_utils


def _convert_grads_to_dtensor(grads, variables, optimizer_state_variables=None):
    """Convert regular torch.Tensor gradients to DTensor to match optimizer states.
    
    When training with DTensor (PyTorch distributed), the optimizer states
    (momentum, velocity) are converted to DTensors via distribute_variable().
    However, gradients remain as regular torch.Tensors. This causes a mismatch
    when torch._foreach_* operations are called, as they cannot handle mixed
    types.
    
    This function converts gradients to DTensors with the same placements as
    the corresponding optimizer state variables.
    
    Args:
        grads: List of gradient tensors
        variables: List of model variable tensors (optional, for backward compat)
        optimizer_state_variables: List of optimizer state variable tensors to check
            for DTensor placement. If provided, these take precedence over 'variables'.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Skip if not in distributed context
    if not torch.distributed.is_initialized():
        return grads
    
    # Import here to avoid circular imports
    from torch.distributed._tensor import DTensor, Replicate
    from keras.src.backend.torch.distribution_lib import is_dtensor
    from keras.src.backend.common import global_state
    
    # CRITICAL FIX: Get the device mesh from the current distribution context,
    # not from global cache. This ensures we use the correct mesh for the
    # current distribution type (DataParallel vs ModelParallel).
    from keras.src.distribution.distribution_lib import distribution, ModelParallel, DataParallel
    
    current_dist = distribution()
    torch_device_mesh = None
    
    # Get the device mesh from the current distribution if available
    if current_dist is not None and hasattr(current_dist, 'device_mesh'):
        from keras.src.backend.torch.distribution_lib import _to_backend_mesh
        torch_device_mesh = _to_backend_mesh(current_dist.device_mesh)
        logger.debug(f"_convert_grads_to_dtensor: got device_mesh from current_dist: {torch_device_mesh}")
    
    if torch_device_mesh is None:
        # Fallback to _get_default_device_mesh for non-distributed cases
        from keras.src.backend.torch.distribution_lib import _get_default_device_mesh
        torch_device_mesh = _get_default_device_mesh()
        logger.debug(f"_convert_grads_to_dtensor: got device_mesh from fallback: {torch_device_mesh}")
    
    # CRITICAL FIX: Even if no mesh from distribution, check if we have a cached mesh
    # This handles the case where distribution scope has exited but we still have
    # DTensors from ModelParallel training
    if torch_device_mesh is None:
        # Check for cached mesh - look for 1D mesh which is used in multi-process MP
        cached_mesh = global_state.get_global_attribute("torch_device_mesh", None)
        if cached_mesh is not None and hasattr(cached_mesh, 'mesh'):
            # Check if it's a 1D mesh (MP in multi-process uses 1D)
            if cached_mesh.mesh.ndim == 1:
                torch_device_mesh = cached_mesh
                logger.debug(f"_convert_grads_to_dtensor: got device_mesh from global cache: {torch_device_mesh}")
    
    if torch_device_mesh is None:
        return grads
    
    logger.debug(f"_convert_grads_to_dtensor: device_mesh={torch_device_mesh}")
    
    # Determine which variables to check for DTensor
    # Priority: optimizer_state_variables > variables
    vars_to_check = optimizer_state_variables if optimizer_state_variables is not None else variables
    
    # Check if any variable is a DTensor
    # In DataParallel, model weights are NOT DTensors but optimizer states ARE
    # So we need to check optimizer_state_variables if provided
    has_dtensor = False
    reference_dtensor = None
    if vars_to_check:
        for i, v in enumerate(vars_to_check):
            # Get the actual tensor value - could be wrapped in a Keras Variable
            value = getattr(v, 'value', v)
            logger.debug(f"_convert_grads_to_dtensor: checking var {i}, type={type(value)}, is_dtensor={isinstance(value, DTensor)}")
            if isinstance(value, DTensor):
                has_dtensor = True
                reference_dtensor = value
                logger.debug(f"_convert_grads_to_dtensor: found DTensor at index {i}, placements={value.placements}")
                break
    
    if not has_dtensor:
        logger.debug("_convert_grads_to_dtensor: no DTensor found in variables, returning grads as-is")
        return grads
    
    # Convert grads to DTensors with the same placements as variables
    converted_grads = []
    for i, (grad, variable) in enumerate(zip(grads, vars_to_check if vars_to_check else grads)):
        if grad is None:
            converted_grads.append(None)
            continue
        
        # CRITICAL: Check if grad is already a DTensor
        # This can happen when the model weights are DTensors and the backward
        # pass automatically creates DTensor gradients
        if isinstance(grad, DTensor):
            logger.debug(f"_convert_grads_to_dtensor: grad {i} is already a DTensor, placements={grad.placements}")
            converted_grads.append(grad)
            continue
            
        # Get the actual tensor value from the optimizer state variable
        value = getattr(variable, 'value', variable)
        if isinstance(value, DTensor):
            # Use the same placements as the optimizer state variable
            placements = value.placements
            logger.debug(f"_convert_grads_to_dtensor: converting grad {i} with placements={placements}")
            dtensor = DTensor.from_local(grad, torch_device_mesh, placements)
            converted_grads.append(dtensor)
        elif reference_dtensor is not None:
            # Use the placements from the reference DTensor
            placements = reference_dtensor.placements
            logger.debug(f"_convert_grads_to_dtensor: converting grad {i} with reference placements={placements}")
            dtensor = DTensor.from_local(grad, torch_device_mesh, placements)
            converted_grads.append(dtensor)
        else:
            # For non-DTensor variables, replicate the gradient
            logger.debug(f"_convert_grads_to_dtensor: keeping grad {i} as-is (non-DTensor variable)")
            converted_grads.append(grad)
    
    logger.debug(f"_convert_grads_to_dtensor: converted {len(converted_grads)} grads")
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
        # Convert grads to DTensors to match accumulated gradients (DTensors)
        # This prevents "aten._foreach_add_.List: got mixed torch.Tensor and DTensor" error
        # Pass acc_grads as optimizer_state_variables to ensure proper DTensor detection
        converted_grads = _convert_grads_to_dtensor(grads, acc_grads, optimizer_state_variables=acc_grads)

        acc_list = [v.value for v in acc_grads]
        torch._foreach_add_(acc_list, converted_grads, alpha=1.0)

