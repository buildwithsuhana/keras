import torch

from keras.src import ops
from keras.src import optimizers
from keras.src.backend.torch.optimizers import torch_parallel_optimizer


def _is_dtensor(tensor):
    """Check if a tensor is a PyTorch DTensor."""
    try:
        from torch.distributed._tensor import DTensor
        return isinstance(tensor, DTensor)
    except ImportError:
        return False


def _ensure_dtensor(tensor, dtensor_ref):
    """Convert a regular torch.Tensor to DTensor matching the reference DTensor."""
    if _is_dtensor(tensor):
        return tensor
    
    # Get the mesh and placements from the reference DTensor
    if _is_dtensor(dtensor_ref):
        from torch.distributed._tensor import distribute_tensor
        return distribute_tensor(tensor, dtensor_ref.device_mesh, dtensor_ref.placements)
    
    return tensor


def _grads_to_dtensor(grads, variables):
    """Convert gradients to DTensors matching the variable layout."""
    if not _is_dtensor(variables[0]):
        return grads
    
    # All variables should have the same mesh and placements
    # Use the first variable as reference
    ref_dtensor = variables[0]
    
    # Convert each gradient to DTensor
    return [_ensure_dtensor(g, ref_dtensor) for g in grads]


class Adam(torch_parallel_optimizer.TorchParallelOptimizer, optimizers.Adam):
    @torch._dynamo.disable
    def _parallel_update_step(
        self,
        grads,
        variables,
        learning_rate,
    ):
        keras_variables = variables
        variables = [v.value for v in variables]

        # Convert grads to DTensors if variables are DTensors
        grads = _grads_to_dtensor(grads, variables)

        # Convert learning rate to native Python scalar for DTensor compatibility
        lr = torch_parallel_optimizer._to_native_scalar(learning_rate)
        
        # Convert iterations to native Python scalar
        local_step = torch_parallel_optimizer._to_native_scalar(self.iterations)
        if hasattr(local_step, 'item'):
            local_step = local_step.item()
        local_step = int(local_step) + 1  # Start from 1

        # Use native Python scalars for beta powers
        beta_1 = torch_parallel_optimizer._to_native_scalar(self.beta_1)
        beta_2 = torch_parallel_optimizer._to_native_scalar(self.beta_2)
        beta_1_power = beta_1 ** local_step
        beta_2_power = beta_2 ** local_step

        # Use native Python scalars for all operations
        one_minus_beta_1_power = 1.0 - beta_1_power
        one_minus_beta_2_power = 1.0 - beta_2_power
        
        # Calculate alpha using native Python operations
        alpha = lr * (one_minus_beta_2_power ** 0.5) / one_minus_beta_1_power

        m_list = [
            self._momentums[self._get_variable_index(variable)].value
            for variable in keras_variables
        ]
        v_list = [
            self._velocities[self._get_variable_index(variable)].value
            for variable in keras_variables
        ]

        # Convert optimizer scalars to native Python scalars for DTensor compatibility
        epsilon = torch_parallel_optimizer._to_native_scalar(self.epsilon)
        one_minus_beta_1 = 1.0 - beta_1
        one_minus_beta_2 = 1.0 - beta_2

        torch._foreach_mul_(m_list, beta_1)
        torch._foreach_add_(m_list, grads, alpha=one_minus_beta_1)

        torch._foreach_mul_(v_list, beta_2)
        torch._foreach_add_(
            v_list, torch._foreach_mul(grads, grads), alpha=one_minus_beta_2
        )

        if self.amsgrad:
            v_hat_list = [
                self._velocity_hats[self._get_variable_index(variable)].value
                for variable in keras_variables
            ]
            torch._foreach_maximum_(v_hat_list, v_list)
            v_list = v_hat_list

        torch._foreach_add_(
            variables,
            torch._foreach_div(
                torch._foreach_mul(m_list, alpha),
                torch._foreach_add(torch._foreach_sqrt(v_list), epsilon),
            ),
            alpha=-1.0,
        )
