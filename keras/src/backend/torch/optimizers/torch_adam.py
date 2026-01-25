import torch

from keras.src import ops
from keras.src import optimizers
from keras.src.backend.torch.optimizers import torch_parallel_optimizer


def _is_dtensor(tensor):
    """Check if a tensor is a PyTorch DTensor."""
    try:
        from torch.distributed._tensor import DTensor
        return isinstance(tensor, DTensor)
    except ImportError:
        return False


class Adam(torch_parallel_optimizer.TorchParallelOptimizer, optimizers.Adam):
    def _parallel_update_step(
        self,
        grads,
        variables,
        learning_rate,
    ):
        keras_variables = variables
        variables = [v.value for v in variables]

        # Debug: Check tensor types
        print(f"[DEBUG Adam] Variables type check:")
        for i, v in enumerate(variables[:3]):  # Check first 3
            print(f"  variables[{i}]: {type(v).__name__}, is_dtensor: {_is_dtensor(v)}")
        print(f"  grads[0]: {type(grads[0]).__name__}, is_dtensor: {_is_dtensor(grads[0])}")

        dtype = variables[0].dtype
        # Use ops helpers to ensure any DTensor interactions go through
        # the backend dispatch (avoids mixing torch.Tensor and DTensor).
        lr = ops.cast(learning_rate, dtype)
        local_step = ops.cast(ops.add(self.iterations, 1), dtype)

        beta_1_power = ops.power(ops.cast(self.beta_1, dtype), local_step)
        beta_2_power = ops.power(ops.cast(self.beta_2, dtype), local_step)

        one = ops.cast(1, dtype)
        numerator = ops.multiply(lr, ops.sqrt(ops.subtract(one, beta_2_power)))
        denominator = ops.subtract(one, beta_1_power)
        alpha = ops.divide(numerator, denominator)

        # Convert alpha to native Python scalar for DTensor compatibility
        alpha = torch_parallel_optimizer._to_native_scalar(alpha)

        m_list = [
            self._momentums[self._get_variable_index(variable)].value
            for variable in keras_variables
        ]
        v_list = [
            self._velocities[self._get_variable_index(variable)].value
            for variable in keras_variables
        ]

        # Debug: Check momentums and velocities types
        print(f"[DEBUG Adam] Internal variables type check:")
        print(f"  m_list[0]: {type(m_list[0]).__name__}, is_dtensor: {_is_dtensor(m_list[0])}")
        print(f"  v_list[0]: {type(v_list[0]).__name__}, is_dtensor: {_is_dtensor(v_list[0])}")

        # Convert optimizer scalars to native Python scalars for DTensor compatibility
        beta_1 = torch_parallel_optimizer._to_native_scalar(self.beta_1)
        beta_2 = torch_parallel_optimizer._to_native_scalar(self.beta_2)
        epsilon = torch_parallel_optimizer._to_native_scalar(self.epsilon)

        torch._foreach_mul_(m_list, beta_1)
        torch._foreach_add_(m_list, grads, alpha=1 - beta_1)

        torch._foreach_mul_(v_list, beta_2)
        torch._foreach_add_(
            v_list, torch._foreach_mul(grads, grads), alpha=1 - beta_2
        )

        if self.amsgrad:
            v_hat_list = [
                self._velocity_hats[self._get_variable_index(variable)].value
                for variable in keras_variables
            ]
            torch._foreach_maximum_(v_hat_list, v_list)
            v_list = v_hat_list

        torch._foreach_add_(
            variables,
            torch._foreach_div(
                torch._foreach_mul(m_list, alpha),
                torch._foreach_add(torch._foreach_sqrt(v_list), epsilon),
            ),
            alpha=-1,
        )
