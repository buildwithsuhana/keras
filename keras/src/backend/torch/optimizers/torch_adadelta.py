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

        dtype = variables[0].dtype
        lr = ops.cast(learning_rate, dtype)
        try:
            from keras.src.backend.torch import core as torch_core
            import numpy as _np

            _lr_val = torch_core.convert_to_numpy(lr)
            if isinstance(_lr_val, _np.ndarray):
                _lr_val = _lr_val.item()
            lr = float(_lr_val)
        except Exception:
            pass
        rho = self.rho

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
        torch._foreach_mul_(accumulated_delta_vars, rho)
        torch._foreach_add_(
            accumulated_delta_vars,
            torch._foreach_mul(delta_vars, delta_vars),
            alpha=1 - rho,
        )

        torch._foreach_add_(variables, delta_vars, alpha=lr)
