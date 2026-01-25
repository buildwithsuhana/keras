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

        dtype = variables[0].dtype
        lr = ops.cast(learning_rate, dtype)

        local_step = ops.cast(ops.add(self.iterations, 1), dtype)

        beta_1_power = ops.power(ops.cast(self.beta_1, dtype), local_step)

        # Pre-compute scalar denominator for foreach calls to avoid mixing
        # DTensor and torch.Tensor when beta_1_power is a DTensor.
        try:
            from keras.src.backend.torch import core as torch_core
            import numpy as _np

            _den = ops.subtract(ops.cast(1, dtype), beta_1_power)
            _den_val = torch_core.convert_to_numpy(_den)
            if isinstance(_den_val, _np.ndarray):
                _den_val = _den_val.item()
            den_scalar = float(_den_val)
        except Exception:
            den_scalar = None

        m_list = [
            self._m[self._get_variable_index(variable)].value
            for variable in keras_variables
        ]
        u_list = [
            self._u[self._get_variable_index(variable)].value
            for variable in keras_variables
        ]

        torch._foreach_mul_(m_list, self.beta_1)
        torch._foreach_add_(m_list, grads, alpha=1 - self.beta_1)

        torch._foreach_mul_(u_list, self.beta_2)
        torch._foreach_maximum_(u_list, torch._foreach_abs(grads))

        # If we could compute a scalar denominator, use it in the foreach call;
        # otherwise fall back to the original expression.
        denom = den_scalar if den_scalar is not None else ops.subtract(ops.cast(1, dtype), beta_1_power)

        torch._foreach_add_(
            variables,
            torch._foreach_div(
                torch._foreach_mul(m_list, lr),
                torch._foreach_mul(
                    torch._foreach_add(u_list, self.epsilon),
                    denom,
                ),
            ),
            alpha=-1,
        )
