import torch

from keras.src.optimizers.base_optimizer import BaseOptimizer
from keras.src.utils import torch_utils


def _to_native_scalar(value):
    """Convert a tensor/DTensor to a native Python scalar.

    This helper is essential for DTensor compatibility. torch._foreach_*
    operations require native Python scalars when interacting with DTensor
    operands. If a DTensor scalar is passed directly, PyTorch will raise
    "aten.sub.Tensor: got mixed torch.Tensor and DTensor" errors.

    Args:
        value: A scalar value (float, int, or torch.Tensor/DTensor)

    Returns:
        A native Python float or int
    """
    if isinstance(value, torch.Tensor):
        # Use convert_to_numpy which properly handles DTensor
        from keras.src.backend.torch.core import convert_to_numpy

        numpy_val = convert_to_numpy(value)
        # Handle numpy array scalars
        if hasattr(numpy_val, "item"):
            return numpy_val.item()
        return numpy_val
    return value


class TorchParallelOptimizer(BaseOptimizer):
    @torch_utils.no_grad
    def _backend_update_step(self, grads, trainable_variables, learning_rate):
        self._parallel_update_step(
            grads,
            trainable_variables,
            learning_rate,
        )

    @torch_utils.no_grad
    def _backend_reset_gradient_accumulators(self):
        acc_list = [
            v.value for v in self._accumulated_gradients if v is not None
        ]
        torch._foreach_mul_(acc_list, 0.0)

    @torch_utils.no_grad
    def _backend_increment_gradient_accumulators(self, grads, acc_grads):
        acc_list = [v.value for v in acc_grads]
        torch._foreach_add_(acc_list, grads, alpha=1.0)
