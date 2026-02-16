import torch

from keras.src import ops
from keras.src import optimizers
from keras.src.backend.torch import core
from keras.src.backend.torch.optimizers import torch_parallel_optimizer


class Nadam(torch_parallel_optimizer.TorchParallelOptimizer, optimizers.Nadam):
    def _parallel_update_step(
        self,
        grads,
        variables,
        learning_rate,
    ):
        keras_variables = variables
        variables = [v.value for v in variables]

        # Get optimizer state variables (momentum and velocity)
        m_list = [
            self._momentums[self._get_variable_index(variable)].value
            for variable in keras_variables
        ]
        v_list = [
            self._velocities[self._get_variable_index(variable)].value
            for variable in keras_variables
        ]
        
        # Combine optimizer state variables to check for DTensor
        optimizer_state_variables = m_list + v_list

        # Convert gradients to DTensor if optimizer states are DTensor
        # This is required for torch._foreach_* operations to work with DTensor
        grads = torch_parallel_optimizer._convert_grads_to_dtensor(
            grads, keras_variables, optimizer_state_variables
        )

        # Check if we're working with DTensors
        from torch.distributed._tensor import DTensor
        
        # CRITICAL FIX: Check if we have a MIXTURE of DTensors and regular tensors
        has_dtensor = any(isinstance(m, DTensor) for m in m_list) if m_list else False
        has_regular_tensor = any(not isinstance(m, DTensor) for m in m_list) if m_list else False
        
        # If we have a MIXTURE of DTensors and regular tensors, convert all to local tensors
        use_dtensor = has_dtensor and not has_regular_tensor
        
        dtype = variables[0].dtype
        lr = ops.cast(learning_rate, dtype)

        local_step = ops.cast(self.iterations + 1, dtype)
        next_step = ops.cast(self.iterations + 2, dtype)
        decay = ops.cast(0.96, dtype)
        beta_1 = ops.cast(self.beta_1, dtype)
        beta_2 = ops.cast(self.beta_2, dtype)
        u_t = beta_1 * (1.0 - 0.5 * (ops.power(decay, local_step)))
        u_t_1 = beta_1 * (1.0 - 0.5 * (ops.power(decay, next_step)))
        u_product_t = self._u_product.value * u_t
        u_product_t_1 = u_product_t * u_t_1
        beta_2_power = ops.power(beta_2, local_step)

        self._u_product.assign(u_product_t)

        # Handle the case where we have mixed DTensors and regular tensors
        if has_dtensor and has_regular_tensor:
            # Convert DTensors to local tensors for the operation
            m_list_local = []
            for m in m_list:
                if isinstance(m, DTensor):
                    m_list_local.append(m.to_local())
                else:
                    m_list_local.append(m)
            v_list_local = []
            for v in v_list:
                if isinstance(v, DTensor):
                    v_list_local.append(v.to_local())
                else:
                    v_list_local.append(v)
            
            # Also convert grads to local if they're DTensors
            grads_local = []
            for g in grads:
                if g is None:
                    grads_local.append(None)
                elif isinstance(g, DTensor):
                    grads_local.append(g.to_local())
                else:
                    grads_local.append(g)
            
            # Also convert variables to local if they're DTensors
            variables_local = []
            for v in variables:
                if isinstance(v, DTensor):
                    variables_local.append(v.to_local())
                else:
                    variables_local.append(v)
            
            # Use local tensors for operations
            torch._foreach_mul_(m_list_local, self.beta_1)
            torch._foreach_add_(m_list_local, grads_local, alpha=1 - self.beta_1)

            torch._foreach_mul_(v_list_local, self.beta_2)
            torch._foreach_add_(
                v_list_local, torch._foreach_mul(grads_local, grads_local), alpha=1 - self.beta_2
            )

            m_hat_list = torch._foreach_add(
                torch._foreach_div(
                    torch._foreach_mul(m_list_local, u_t_1),
                    1 - core.convert_to_numpy(u_product_t_1),
                ),
                torch._foreach_div(
                    torch._foreach_mul(grads_local, 1 - u_t),
                    1 - core.convert_to_numpy(u_product_t),
                ),
            )

            v_hat_list = torch._foreach_div(v_list_local, 1 - beta_2_power)

            torch._foreach_add_(
                variables_local,
                torch._foreach_div(
                    torch._foreach_mul(m_hat_list, lr),
                    torch._foreach_add(
                        torch._foreach_sqrt(v_hat_list), self.epsilon
                    ),
                ),
                alpha=-1,
            )
            
            # Copy values back to original variables (DTensors)
            for i, (v_local, v_orig) in enumerate(zip(variables_local, variables)):
                if isinstance(v_orig, DTensor):
                    placements = v_orig.placements
                    variables[i] = v_orig.from_local(v_local, v_orig.device_mesh, placements, requires_grad=v_orig.requires_grad)
                    
        elif use_dtensor:
            # All are DTensors - convert scalars to tensors
            beta_1_tensor = torch.tensor(self.beta_1, dtype=dtype)
            beta_2_tensor = torch.tensor(self.beta_2, dtype=dtype)
            one_minus_beta_1 = torch.tensor(1 - self.beta_1, dtype=dtype)
            one_minus_beta_2 = torch.tensor(1 - self.beta_2, dtype=dtype)
            
            torch._foreach_mul_(m_list, beta_1_tensor)
            torch._foreach_add_(m_list, grads, alpha=one_minus_beta_1)

            torch._foreach_mul_(v_list, beta_2_tensor)
            torch._foreach_add_(
                v_list, torch._foreach_mul(grads, grads), alpha=one_minus_beta_2
            )

            m_hat_list = torch._foreach_add(
                torch._foreach_div(
                    torch._foreach_mul(m_list, u_t_1),
                    1 - core.convert_to_numpy(u_product_t_1),
                ),
                torch._foreach_div(
                    torch._foreach_mul(grads, 1 - u_t),
                    1 - core.convert_to_numpy(u_product_t),
                ),
            )

            v_hat_list = torch._foreach_div(v_list, 1 - beta_2_power)

            torch._foreach_add_(
                variables,
                torch._foreach_div(
                    torch._foreach_mul(m_hat_list, lr),
                    torch._foreach_add(
                        torch._foreach_sqrt(v_hat_list), self.epsilon
                    ),
                ),
                alpha=-1,
            )
        else:
            # No DTensors - use regular tensor operations
            torch._foreach_mul_(m_list, self.beta_1)
            torch._foreach_add_(m_list, grads, alpha=1 - self.beta_1)

            torch._foreach_mul_(v_list, self.beta_2)
            torch._foreach_add_(
                v_list, torch._foreach_mul(grads, grads), alpha=1 - self.beta_2
            )

            m_hat_list = torch._foreach_add(
                torch._foreach_div(
                    torch._foreach_mul(m_list, u_t_1),
                    1 - core.convert_to_numpy(u_product_t_1),
                ),
                torch._foreach_div(
                    torch._foreach_mul(grads, 1 - u_t),
                    1 - core.convert_to_numpy(u_product_t),
                ),
            )

            v_hat_list = torch._foreach_div(v_list, 1 - beta_2_power)

            torch._foreach_add_(
                variables,
                torch._foreach_div(
                    torch._foreach_mul(m_hat_list, lr),
                    torch._foreach_add(
                        torch._foreach_sqrt(v_hat_list), self.epsilon
                    ),
                ),
                alpha=-1,
            )
