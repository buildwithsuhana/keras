import torch

from keras.src import ops
from keras.src import optimizers
from keras.src.backend.torch.optimizers import torch_parallel_optimizer


class Lion(torch_parallel_optimizer.TorchParallelOptimizer, optimizers.Lion):
    def _parallel_update_step(
        self,
        grads,
        variables,
        learning_rate,
    ):
        keras_variables = variables
        variables = [v.value for v in variables]

        # Get optimizer state variables (momentum)
        m_list = [
            self._momentums[self._get_variable_index(variable)].value
            for variable in keras_variables
        ]
        
        # Combine optimizer state variables to check for DTensor
        optimizer_state_variables = m_list

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

        # Handle the case where we have mixed DTensors and regular tensors
        if has_dtensor and has_regular_tensor:
            # Convert DTensors to local tensors for the operation
            m_list_local = []
            for m in m_list:
                if isinstance(m, DTensor):
                    m_list_local.append(m.to_local())
                else:
                    m_list_local.append(m)
            
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
            c_t = torch._foreach_mul(m_list_local, self.beta_1)
            torch._foreach_add_(c_t, grads_local, alpha=1 - self.beta_1)
            c_t = [c.sign() for c in c_t]

            torch._foreach_add_(
                variables_local,
                torch._foreach_mul(c_t, lr),
                alpha=-1,
            )

            torch._foreach_mul_(m_list_local, self.beta_2)
            torch._foreach_add_(m_list_local, grads_local, alpha=1 - self.beta_2)
            
            # Copy values back to original variables (DTensors)
            for i, (v_local, v_orig) in enumerate(zip(variables_local, variables)):
                if isinstance(v_orig, DTensor):
                    placements = v_orig.placements
                    # Set requires_grad on the local tensor before creating DTensor
                    v_local = v_local.requires_grad_(v_orig.requires_grad)
                    variables[i] = v_orig.from_local(v_local, v_orig.device_mesh, placements)
                    
        elif use_dtensor:
            # All are DTensors - convert scalars to tensors
            one_minus_beta_1 = torch.tensor(1 - self.beta_1, dtype=dtype)
            beta_2_tensor = torch.tensor(self.beta_2, dtype=dtype)
            one_minus_beta_2 = torch.tensor(1 - self.beta_2, dtype=dtype)
            
            c_t = torch._foreach_mul(m_list, beta_2_tensor)  # Use beta_2 for consistency in sign calc
            torch._foreach_add_(c_t, grads, alpha=one_minus_beta_1)
            c_t = [c.sign() for c in c_t]

            torch._foreach_add_(
                variables,
                torch._foreach_mul(c_t, lr),
                alpha=-1,
            )

            torch._foreach_mul_(m_list, beta_2_tensor)
            torch._foreach_add_(m_list, grads, alpha=one_minus_beta_2)
        else:
            # No DTensors - use regular tensor operations
            c_t = torch._foreach_mul(m_list, self.beta_1)
            torch._foreach_add_(c_t, grads, alpha=1 - self.beta_1)
            c_t = [c.sign() for c in c_t]

            torch._foreach_add_(
                variables,
                torch._foreach_mul(c_t, lr),
                alpha=-1,
            )

            torch._foreach_mul_(m_list, self.beta_2)
            torch._foreach_add_(m_list, grads, alpha=1 - self.beta_2)
