import torch

from keras.src import ops
from keras.src import optimizers
from keras.src.backend.torch.optimizers import torch_parallel_optimizer


class RMSprop(
    torch_parallel_optimizer.TorchParallelOptimizer, optimizers.RMSprop
):
    def _parallel_update_step(
        self,
        grads,
        variables,
        learning_rate,
    ):
        keras_variables = variables
        variables = [v.value for v in variables]

        # Get optimizer state variables (velocities, and optionally average_grads and momentum)
        velocities = [
            self._velocities[self._get_variable_index(variable)].value
            for variable in keras_variables
        ]
        
        # Combine optimizer state variables to check for DTensor
        optimizer_state_variables = velocities[:]

        # CRITICAL: Convert gradients to DTensor FIRST, before any operations
        # This is required for torch._foreach_* operations to work with DTensor
        # Must be done before any torch._foreach_* call that uses grads
        grads = torch_parallel_optimizer._convert_grads_to_dtensor(
            grads, keras_variables, optimizer_state_variables
        )
        
        # Check if we're working with DTensors
        from torch.distributed._tensor import DTensor
        
        # CRITICAL FIX: Check if we have a MIXTURE of DTensors and regular tensors
        has_dtensor = any(isinstance(v, DTensor) for v in velocities) if velocities else False
        has_regular_tensor = any(not isinstance(v, DTensor) for v in velocities) if velocities else False
        
        # If we have a MIXTURE of DTensors and regular tensors, convert all to local tensors
        use_dtensor = has_dtensor and not has_regular_tensor
        
        dtype = variables[0].dtype
        lr = ops.cast(learning_rate, dtype)

        rho = self.rho

        # Handle the case where we have mixed DTensors and regular tensors
        if has_dtensor and has_regular_tensor:
            # Convert DTensors to local tensors for the operation
            velocities_local = []
            for v in velocities:
                if isinstance(v, DTensor):
                    velocities_local.append(v.to_local())
                else:
                    velocities_local.append(v)
            
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
            torch._foreach_mul_(velocities_local, rho)
            torch._foreach_add_(
                velocities_local, torch._foreach_mul(grads_local, grads_local), alpha=1 - rho
            )

            denominators = torch._foreach_add(velocities_local, self.epsilon)
            if self.centered:
                average_grads = [
                    self._average_gradients[
                        self._get_variable_index(variable)
                    ].value
                    for variable in keras_variables
                ]
                
                # Convert to local
                average_grads_local = []
                for a in average_grads:
                    if isinstance(a, DTensor):
                        average_grads_local.append(a.to_local())
                    else:
                        average_grads_local.append(a)
                
                torch._foreach_mul_(average_grads_local, rho)
                torch._foreach_add_(average_grads_local, grads_local, alpha=1 - rho)
                torch._foreach_add_(
                    denominators,
                    torch._foreach_mul(average_grads_local, average_grads_local),
                    alpha=-1,
                )
            torch._foreach_sqrt_(denominators)
            increments = torch._foreach_div(
                torch._foreach_mul(grads_local, lr), denominators
            )

            if self.momentum > 0:
                momentum_list = [
                    self._momentums[self._get_variable_index(variable)].value
                    for variable in keras_variables
                ]
                
                # Convert to local
                momentum_list_local = []
                for m in momentum_list:
                    if isinstance(m, DTensor):
                        momentum_list_local.append(m.to_local())
                    else:
                        momentum_list_local.append(m)
                
                torch._foreach_mul_(momentum_list_local, self.momentum)
                torch._foreach_add_(momentum_list_local, increments)
                torch._foreach_add_(variables_local, momentum_list_local, alpha=-1)
            else:
                torch._foreach_add_(variables_local, increments, alpha=-1)
            
            # Copy values back to original variables (DTensors)
            for i, (v_local, v_orig) in enumerate(zip(variables_local, variables)):
                if isinstance(v_orig, DTensor):
                    placements = v_orig.placements
                    variables[i] = v_orig.from_local(v_local, v_orig.device_mesh, placements, requires_grad=v_orig.requires_grad)
                    
        elif use_dtensor:
            # All are DTensors - convert scalars to tensors
            rho_tensor = torch.tensor(rho, dtype=dtype)
            one_minus_rho = torch.tensor(1 - rho, dtype=dtype)
            
            torch._foreach_mul_(velocities, rho_tensor)
            torch._foreach_add_(
                velocities, torch._foreach_mul(grads, grads), alpha=one_minus_rho
            )

            denominators = torch._foreach_add(velocities, self.epsilon)
            if self.centered:
                average_grads = [
                    self._average_gradients[
                        self._get_variable_index(variable)
                    ].value
                    for variable in keras_variables
                ]
                optimizer_state_variables.extend(average_grads)
                
                # Check again for DTensor since average_grads might be different
                use_dtensor = use_dtensor or any(isinstance(a, DTensor) for a in average_grads) if average_grads else use_dtensor
                
                if use_dtensor:
                    torch._foreach_mul_(average_grads, rho_tensor)
                    torch._foreach_add_(average_grads, grads, alpha=one_minus_rho)
                else:
                    torch._foreach_mul_(average_grads, rho)
                    torch._foreach_add_(average_grads, grads, alpha=1 - rho)
                torch._foreach_add_(
                    denominators,
                    torch._foreach_mul(average_grads, average_grads),
                    alpha=-1,
                )
            torch._foreach_sqrt_(denominators)
            increments = torch._foreach_div(
                torch._foreach_mul(grads, lr), denominators
            )

            if self.momentum > 0:
                momentum_list = [
                    self._momentums[self._get_variable_index(variable)].value
                    for variable in keras_variables
                ]
                optimizer_state_variables.extend(momentum_list)
                
                # Check again for DTensor since momentum_list might be different
                use_dtensor = use_dtensor or any(isinstance(m, DTensor) for m in momentum_list) if momentum_list else use_dtensor
                
                if use_dtensor:
                    momentum_tensor = torch.tensor(self.momentum, dtype=dtype)
                    torch._foreach_mul_(momentum_list, momentum_tensor)
                    torch._foreach_add_(momentum_list, increments)
                    torch._foreach_add_(variables, momentum_list, alpha=-1)
                else:
                    torch._foreach_mul_(momentum_list, self.momentum)
                    torch._foreach_add_(momentum_list, increments)
                    torch._foreach_add_(variables, momentum_list, alpha=-1)
            else:
                torch._foreach_add_(variables, increments, alpha=-1)
        else:
            # No DTensors - use regular tensor operations
            torch._foreach_mul_(velocities, rho)
            torch._foreach_add_(
                velocities, torch._foreach_mul(grads, grads), alpha=1 - rho
            )

            denominators = torch._foreach_add(velocities, self.epsilon)
            if self.centered:
                average_grads = [
                    self._average_gradients[
                        self._get_variable_index(variable)
                    ].value
                    for variable in keras_variables
                ]
                optimizer_state_variables.extend(average_grads)
                
                # Check again for DTensor since average_grads might be different
                use_dtensor = use_dtensor or any(isinstance(a, DTensor) for a in average_grads) if average_grads else use_dtensor
                
                if use_dtensor:
                    torch._foreach_mul_(average_grads, rho_tensor)
                    torch._foreach_add_(average_grads, grads, alpha=one_minus_rho)
                else:
                    torch._foreach_mul_(average_grads, rho)
                    torch._foreach_add_(average_grads, grads, alpha=1 - rho)
                torch._foreach_add_(
                    denominators,
                    torch._foreach_mul(average_grads, average_grads),
                    alpha=-1,
                )
            torch._foreach_sqrt_(denominators)
            increments = torch._foreach_div(
                torch._foreach_mul(grads, lr), denominators
            )

            if self.momentum > 0:
                momentum_list = [
                    self._momentums[self._get_variable_index(variable)].value
                    for variable in keras_variables
                ]
                optimizer_state_variables.extend(momentum_list)
                
                # Check again for DTensor since momentum_list might be different
                use_dtensor = use_dtensor or any(isinstance(m, DTensor) for m in momentum_list) if momentum_list else use_dtensor
                
                if use_dtensor:
                    momentum_tensor = torch.tensor(self.momentum, dtype=dtype)
                    torch._foreach_mul_(momentum_list, momentum_tensor)
                    torch._foreach_add_(momentum_list, increments)
                    torch._foreach_add_(variables, momentum_list, alpha=-1)
                else:
                    torch._foreach_mul_(momentum_list, self.momentum)
                    torch._foreach_add_(momentum_list, increments)
                    torch._foreach_add_(variables, momentum_list, alpha=-1)
            else:
                torch._foreach_add_(variables, increments, alpha=-1)
