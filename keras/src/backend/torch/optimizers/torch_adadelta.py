import torch

from keras.src import ops
from keras.src import optimizers
from keras.src.backend.torch.optimizers import torch_parallel_optimizer


class Adadelta(
    torch_parallel_optimizer.TorchParallelOptimizer, optimizers.Adadelta
):
    def _parallel_update_step(
        self,
        grads,
        variables,
        learning_rate,
    ):
        keras_variables = variables
        variables = [v.value for v in variables]

        # Get optimizer state variables (accumulated_grads and accumulated_delta_vars)
        accumulated_grads = [
            self._accumulated_grads[self._get_variable_index(variable)].value
            for variable in keras_variables
        ]
        accumulated_delta_vars = [
            self._accumulated_delta_vars[
                self._get_variable_index(variable)
            ].value
            for variable in keras_variables
        ]
        
        # Combine optimizer state variables to check for DTensor
        optimizer_state_variables = accumulated_grads + accumulated_delta_vars

        # Convert gradients to DTensor if optimizer states are DTensor
        # This is required for torch._foreach_* operations to work with DTensor
        grads = torch_parallel_optimizer._convert_grads_to_dtensor(
            grads, keras_variables, optimizer_state_variables
        )

        # Check if we're working with DTensors
        from torch.distributed._tensor import DTensor
        use_dtensor = any(isinstance(a, DTensor) for a in accumulated_grads) if accumulated_grads else False
        
        dtype = variables[0].dtype
        lr = ops.cast(learning_rate, dtype)
        rho = self.rho

        # CRITICAL FIX: For DTensor operations, scalars must be converted to tensors
        if use_dtensor:
            rho_tensor = torch.tensor(rho, dtype=dtype)
            one_minus_rho = torch.tensor(1 - rho, dtype=dtype)
            
            torch._foreach_mul_(accumulated_grads, rho_tensor)
            torch._foreach_add_(
                accumulated_grads, torch._foreach_mul(grads, grads), alpha=one_minus_rho
            )
        else:
            torch._foreach_mul_(accumulated_grads, rho)
            torch._foreach_add_(
                accumulated_grads, torch._foreach_mul(grads, grads), alpha=1 - rho
            )

        def rms(x):
            return torch._foreach_sqrt(torch._foreach_add(x, self.epsilon))

        delta_vars = torch._foreach_mul(
            torch._foreach_div(
                torch._foreach_mul(rms(accumulated_delta_vars), grads),
                rms(accumulated_grads),
            ),
            -1,
        )
        
        if use_dtensor:
            torch._foreach_mul_(accumulated_delta_vars, rho_tensor)
            torch._foreach_add_(
                accumulated_delta_vars,
                torch._foreach_mul(delta_vars, delta_vars),
                alpha=one_minus_rho,
            )
        else:
            torch._foreach_mul_(accumulated_delta_vars, rho)
            torch._foreach_add_(
                accumulated_delta_vars,
                torch._foreach_mul(delta_vars, delta_vars),
                alpha=1 - rho,
            )

        torch._foreach_add_(variables, delta_vars, alpha=lr)
