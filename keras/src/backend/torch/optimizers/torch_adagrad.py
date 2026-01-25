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

        dtype = variables[0].dtype
        lr = ops.cast(learning_rate, dtype)
        # Convert lr to Python scalar if it's a tensor/DTensor so foreach
        # calls receive a native scalar and avoid DTensor/torch.Tensor mixing.
        try:
            from keras.src.backend.torch import core as torch_core
            import numpy as _np

            _lr_val = torch_core.convert_to_numpy(lr)
            if isinstance(_lr_val, _np.ndarray):
                _lr_val = _lr_val.item()
            lr = float(_lr_val)
        except Exception:
            pass

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
