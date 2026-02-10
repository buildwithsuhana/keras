import torch

from keras.src import ops
from keras.src import optimizers
from keras.src.backend.torch.optimizers import torch_parallel_optimizer


class Adagrad(
    torch_parallel_optimizer.TorchParallelOptimizer, optimizers.Adagrad
):
    def _parallel_update_step(
        self,
        grads,
        variables,
        learning_rate,
    ):
        keras_variables = variables
        variables = [v.value for v in variables]

        # Convert gradients to DTensor if optimizer states are DTensor
        # This is required for torch._foreach_* operations to work with DTensor
        grads = torch_parallel_optimizer._convert_grads_to_dtensor(
            grads, keras_variables
        )

        dtype = variables[0].dtype
        lr = ops.cast(learning_rate, dtype)

        accumulators = [
            self._accumulators[self._get_variable_index(variable)].value
            for variable in keras_variables
        ]
        torch._foreach_add_(accumulators, torch._foreach_mul(grads, grads))
        torch._foreach_add_(
            variables,
            torch._foreach_div(
                torch._foreach_mul(grads, lr),
                torch._foreach_sqrt(
                    torch._foreach_add(accumulators, self.epsilon)
                ),
            ),
            alpha=-1,
        )

