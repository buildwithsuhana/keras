import torch

from keras.src import optimizers
from keras.src.backend.torch.optimizers import torch_parallel_optimizer


class SGD(torch_parallel_optimizer.TorchParallelOptimizer, optimizers.SGD):
    def _parallel_update_step(
        self,
        grads,
        variables,
        learning_rate,
    ):
        keras_variables = variables
        variables = [v.value for v in variables]
        
        # Get optimizer state variables (momentum buffers)
        bufs = None
        if self.momentum != 0:
            bufs = [
                self.momentums[self._get_variable_index(variable)].value
                for variable in keras_variables
            ]
            
            # Combine optimizer state variables to check for DTensor
            optimizer_state_variables = [b for b in bufs if b is not None]

            # Convert gradients to DTensor if optimizer states are DTensor
            # This is required for torch._foreach_* operations to work with DTensor
            grads = torch_parallel_optimizer._convert_grads_to_dtensor(
                grads, keras_variables, optimizer_state_variables
            )

            for i in range(len(bufs)):
                if bufs[i] is None:
                    bufs[i] = torch.clone(grads[i]).detach()

            # Check if we're working with DTensors
            from torch.distributed._tensor import DTensor
            
            # CRITICAL FIX: Check if we have a MIXTURE of DTensors and regular tensors
            has_dtensor = any(isinstance(b, DTensor) for b in bufs if b is not None) if bufs else False
            has_regular_tensor = any(not isinstance(b, DTensor) for b in bufs if b is not None) if bufs else False
            
            # Handle the case where we have mixed DTensors and regular tensors
            if has_dtensor and has_regular_tensor:
                # Convert DTensors to local tensors for the operation
                bufs_local = []
                for b in bufs:
                    if b is None:
                        bufs_local.append(None)
                    elif isinstance(b, DTensor):
                        bufs_local.append(b.to_local())
                    else:
                        bufs_local.append(b)
                
                # Also convert grads to local if they're DTensors
                grads_local = []
                for g in grads:
                    if g is None:
                        grads_local.append(None)
                    elif isinstance(g, DTensor):
                        grads_local.append(g.to_local())
                    else:
                        grads_local.append(g)
                
                variables_local = []
                for v in variables:
                    if isinstance(v, DTensor):
                        variables_local.append(v.to_local())
                    else:
                        variables_local.append(v)
                
                # Use local tensors for operations
                torch._foreach_mul_(bufs_local, self.momentum)
                torch._foreach_add_(bufs_local, grads_local, alpha=-learning_rate)

                if self.nesterov:
                    torch._foreach_add_(variables_local, grads_local, alpha=-learning_rate)
                    torch._foreach_add_(variables_local, bufs_local, alpha=self.momentum)
                else:
                    torch._foreach_add_(variables_local, bufs_local)
                
                # Copy values back to original variables (DTensors)
                for i, (v_local, v_orig) in enumerate(zip(variables_local, variables)):
                    if isinstance(v_orig, DTensor):
                        placements = v_orig.placements
                        # Set requires_grad on the local tensor before creating DTensor
                        v_local = v_local.requires_grad_(v_orig.requires_grad)
                        variables[i] = v_orig.from_local(v_local, v_orig.device_mesh, placements)
                        
            elif has_dtensor:
                # All are DTensors - convert scalars to tensors
                momentum_tensor = torch.tensor(self.momentum, dtype=grads[0].dtype if grads else variables[0].dtype)
                lr_tensor = torch.tensor(-learning_rate, dtype=variables[0].dtype)
                
                torch._foreach_mul_(bufs, momentum_tensor)
                torch._foreach_add_(bufs, grads, alpha=lr_tensor)

                if self.nesterov:
                    torch._foreach_add_(variables, grads, alpha=lr_tensor)
                    torch._foreach_add_(variables, bufs, alpha=momentum_tensor)
                else:
                    torch._foreach_add_(variables, bufs)
            else:
                # No DTensors - use regular tensor operations
                torch._foreach_mul_(bufs, self.momentum)
                torch._foreach_add_(bufs, grads, alpha=-learning_rate)

                if self.nesterov:
                    torch._foreach_add_(variables, grads, alpha=-learning_rate)
                    torch._foreach_add_(variables, bufs, alpha=self.momentum)
                else:
                    torch._foreach_add_(variables, bufs)

        else:
            torch._foreach_add_(variables, grads, alpha=-learning_rate)
