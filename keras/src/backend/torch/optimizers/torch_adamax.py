import torch

from keras.src import ops
from keras.src import optimizers
from keras.src.backend.torch.optimizers import torch_parallel_optimizer


class Adamax(
    torch_parallel_optimizer.TorchParallelOptimizer, optimizers.Adamax
):
    def _parallel_update_step(
        self,
        grads,
        variables,
        learning_rate,
    ):
        keras_variables = variables
        variables = [v.value for v in variables]

        # Get optimizer state variables (m and u)
        m_list = [
            self._m[self._get_variable_index(variable)].value
            for variable in keras_variables
        ]
        u_list = [
            self._u[self._get_variable_index(variable)].value
            for variable in keras_variables
        ]
        
        # Combine optimizer state variables to check for DTensor
        optimizer_state_variables = m_list + u_list

        # Convert gradients to DTensor if optimizer states are DTensor
        # This is required for torch._foreach_* operations to work with DTensor
        grads = torch_parallel_optimizer._convert_grads_to_dtensor(
            grads, keras_variables, optimizer_state_variables
        )

        # Check if we're working with DTensors
        from torch.distributed._tensor import DTensor
        use_dtensor = any(isinstance(m, DTensor) for m in m_list) if m_list else False
        
        dtype = variables[0].dtype
        lr = ops.cast(learning_rate, dtype)

        local_step = ops.cast(self.iterations + 1, dtype)

        beta_1_power = ops.power(ops.cast(self.beta_1, dtype), local_step)

        # CRITICAL FIX: For DTensor operations, scalars must be converted to tensors
        if use_dtensor:
            beta_1_tensor = torch.tensor(self.beta_1, dtype=dtype)
            beta_2_tensor = torch.tensor(self.beta_2, dtype=dtype)
            one_minus_beta_1 = torch.tensor(1 - self.beta_1, dtype=dtype)
            
            torch._foreach_mul_(m_list, beta_1_tensor)
            torch._foreach_add_(m_list, grads, alpha=one_minus_beta_1)

            torch._foreach_mul_(u_list, beta_2_tensor)
            torch._foreach_maximum_(u_list, torch._foreach_abs(grads))
        else:
            torch._foreach_mul_(m_list, self.beta_1)
            torch._foreach_add_(m_list, grads, alpha=1 - self.beta_1)

            torch._foreach_mul_(u_list, self.beta_2)
            torch._foreach_maximum_(u_list, torch._foreach_abs(grads))

        torch._foreach_add_(
            variables,
            torch._foreach_div(
                torch._foreach_mul(m_list, lr),
                torch._foreach_mul(
                    torch._foreach_add(u_list, self.epsilon),
                    1 - beta_1_power,
                ),
            ),
            alpha=-1,
        )
