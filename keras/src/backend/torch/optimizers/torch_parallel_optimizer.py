import torch

from keras.src.backend.torch.distribution_logger import get_logger
from keras.src.optimizers.base_optimizer import BaseOptimizer
from keras.src.utils import torch_utils


class TorchParallelOptimizer(BaseOptimizer):
    @torch_utils.no_grad
    def _backend_update_step(self, grads, trainable_variables, learning_rate):
        logger = get_logger()
        rank = 0  # Will be updated in _parallel_update_step
        
        # Get optimizer name
        optimizer_name = self.__class__.__name__
        
        # Count variables
        num_variables = len(trainable_variables)
        
        logger.info(f"{optimizer_name}: Starting parallel update step - "
                    f"num_variables={num_variables}, learning_rate={learning_rate}")
        
        self._parallel_update_step(
            grads,
            trainable_variables,
            learning_rate,
        )
        
        logger.debug(f"{optimizer_name}: Parallel update step complete")

    @torch_utils.no_grad
    def _backend_reset_gradient_accumulators(self):
        logger = get_logger()
        
        acc_list = [
            v.value for v in self._accumulated_gradients if v is not None
        ]
        
        logger.debug(f"TorchParallelOptimizer: Resetting {len(acc_list)} gradient accumulators")
        
        torch._foreach_mul_(acc_list, 0.0)

    @torch_utils.no_grad
    def _backend_increment_gradient_accumulators(self, grads, acc_grads):
        logger = get_logger()
        
        acc_list = [v.value for v in acc_grads]
        
        logger.debug(f"TorchParallelOptimizer: Incrementing {len(acc_list)} gradient accumulators")
        
        torch._foreach_add_(acc_list, grads, alpha=1.0)
