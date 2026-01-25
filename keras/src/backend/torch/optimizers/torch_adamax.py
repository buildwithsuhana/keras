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
        # Convert lr to native Python scalar for DTensor compatibility
        lr = torch_parallel_optimizer._to_native_scalar(lr)
        # Convert optimizer scalars to native Python scalars for DTensor compatibility
        beta_1 = torch_parallel_optimizer._to_native_scalar(self.beta_1)
        beta_2 = torch_parallel_optimizer._to_native_scalar(self.beta_2)
        epsilon = torch_parallel_optimizer._to_native_scalar(self.epsilon)

        local_step = ops.cast(ops.add(self.iterations, 1), dtype)

        beta_1_power = ops.power(ops.cast(beta_1, dtype), local_step)

        # Pre-compute scalar denominator for DTensor compatibility
        den_scalar = torch_parallel_optimizer._to_native_scalar(
            ops.subtract(ops.cast(1, dtype), beta_1_power)
        )

        m_list = [
            self._m[self._get_variable_index(variable)].value
            for variable in keras_variables
        ]
        u_list = [
            self._u[self._get_variable_index(variable)].value
            for variable in keras_variables
        ]

        torch._foreach_mul_(m_list, beta_1)
        torch._foreach_add_(m_list, grads, alpha=1 - beta_1)

        torch._foreach_mul_(u_list, beta_2)
        torch._foreach_maximum_(u_list, torch._foreach_abs(grads))

        # If we could compute a scalar denominator, use it in the foreach call;
        # otherwise fall back to the original expression.
        denom = den_scalar if den_scalar is not None else ops.subtract(ops.cast(1, dtype), beta_1_power)

        torch._foreach_add_(
            variables,
            torch._foreach_div(
                torch._foreach_mul(m_list, lr),
                torch._foreach_mul(
                    torch._foreach_add(u_list, epsilon),
                    denom,
                ),
            ),
            alpha=-1,
        )
