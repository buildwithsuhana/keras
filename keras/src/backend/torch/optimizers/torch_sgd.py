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
            use_dtensor = any(isinstance(b, DTensor) for b in bufs if b is not None) if bufs else False
            
            # CRITICAL FIX: For DTensor operations, scalars must be converted to tensors
            if use_dtensor:
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
                torch._foreach_mul_(bufs, self.momentum)
                torch._foreach_add_(bufs, grads, alpha=-learning_rate)

                if self.nesterov:
                    torch._foreach_add_(variables, grads, alpha=-learning_rate)
                    torch._foreach_add_(variables, bufs, alpha=self.momentum)
                else:
                    torch._foreach_add_(variables, bufs)

        else:
            torch._foreach_add_(variables, grads, alpha=-learning_rate)
