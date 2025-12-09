import re
from typing import Any

import numpy as np
import keras
from keras.src import ops
from keras.src import optimizers

class CoordinatedOptimizer:
    """Manages an optimizer's state for distributed training."""

    def __init__(
        self,
        base_optimizer: optimizers.Optimizer,
        device_count: int,
        shard_optimizer_states: bool = True,
        tensor_parallel_config=None,
    ):
        self.base_optimizer = base_optimizer
        self.device_count = device_count
        self.shard_optimizer_states = shard_optimizer_states
        self.tensor_parallel_config = tensor_parallel_config
        self.sharded_states = {}

    def _initialize_sharded_states(self):
        return

    def apply_gradients(
        self, gradients_and_vars: list[list[tuple]], shard_models: list
    ):
        synchronized_gradients = self._synchronize_gradients(gradients_and_vars)

        for shard_idx in range(self.device_count):
            shard_opt = shard_models[shard_idx].optimizer
            if hasattr(shard_opt, "inner_optimizer"): 
                shard_opt = shard_opt.inner_optimizer
            elif hasattr(shard_opt, "base_optimizer"):
                shard_opt = shard_opt.base_optimizer

            shard_grads_and_vars = synchronized_gradients[shard_idx]
            shard_opt.apply_gradients(shard_grads_and_vars)

    def _synchronize_gradients(
        self, gradients_and_vars: list[list[tuple]]
    ) -> list[list[tuple]]:
        if not self.tensor_parallel_config:
            return gradients_and_vars

        rules = self.tensor_parallel_config.state_rules.items()
        column_parallel_patterns = {
            pattern
            for pattern, action in rules
            if hasattr(action, "sharding_type")
            and action.sharding_type == "column"
        }

        if not column_parallel_patterns:
            return gradients_and_vars

        num_weights = len(gradients_and_vars[0])
        for i in range(num_weights):
            variable = gradients_and_vars[0][i][1]
            var_name = getattr(variable, "path", getattr(variable, "name", ""))

            if any(re.search(pattern, var_name) for pattern in column_parallel_patterns):
                grads_to_reduce = [
                    g_and_v[i][0]
                    for g_and_v in gradients_and_vars
                    if g_and_v[i][0] is not None
                ]
                if grads_to_reduce:
                    # FIX: Use iterative accumulation on the target device to avoid CPU stacking
                    import keras
                    target_device = grads_to_reduce[0].device
                    
                    with keras.device(target_device):
                        # Start with the first gradient
                        total_grad = ops.convert_to_tensor(grads_to_reduce[0])
                        
                        # Add the rest (implicitly or explicitly moves them to target_device)
                        for g in grads_to_reduce[1:]:
                            total_grad = ops.add(total_grad, ops.convert_to_tensor(g))
                        
                        # Compute mean
                        mean_grad = ops.divide(total_grad, len(grads_to_reduce))
                    
                    for shard_idx in range(self.device_count):
                        if gradients_and_vars[shard_idx][i][0] is not None:
                            gradients_and_vars[shard_idx][i] = (mean_grad, variable)
        return gradients_and_vars

    def get_weights(self) -> list[np.ndarray]:
        return []

    def set_weights(self, weights: list[np.ndarray]):
        pass

    def enable_optimizer_state_sharding(self, variables: list):
        self.shard_optimizer_states = True
        pass


class TensorParallelOptimizer(optimizers.Optimizer):
    def __init__(
        self,
        base_optimizer: optimizers.Optimizer,
        device_count: int,
        tensor_parallel_config=None,
    ):
        if isinstance(base_optimizer, str):
            base_optimizer_instance = optimizers.get(base_optimizer)
        else:
            base_optimizer_instance = base_optimizer

        learning_rate = base_optimizer_instance.learning_rate
        try:
            lr_value = float(ops.convert_to_numpy(learning_rate))
        except:
            lr_value = 0.001

        super().__init__(
            learning_rate=lr_value,
            name=f"TensorParallel_{base_optimizer_instance.name}",
        )

        self.base_optimizer = base_optimizer_instance
        self.device_count = device_count
        self.coordinated_optimizer = CoordinatedOptimizer(
            self.base_optimizer,
            device_count,
            tensor_parallel_config=tensor_parallel_config,
        )

    def apply_gradients(self, grads_and_vars: list, **kwargs):
        is_sharded_grads = (
            isinstance(grads_and_vars, list)
            and grads_and_vars
            and isinstance(grads_and_vars[0], list)
        )
        if is_sharded_grads:
            shard_models = kwargs.get("shard_models")
            self.coordinated_optimizer.apply_gradients(
                grads_and_vars, shard_models
            )
        else:
            self.base_optimizer.apply_gradients(grads_and_vars)

    def update_step(self, gradient, variable, learning_rate):
        return self.base_optimizer.update_step(gradient, variable, learning_rate)

    def build(self, variables: list):
        if self.built: return
        self.coordinated_optimizer.enable_optimizer_state_sharding(variables)
        self.built = True

    def get_config(self):
        return {
            "base_optimizer": keras.optimizers.serialize(self.base_optimizer),
            "device_count": self.device_count,
        }

    @classmethod
    def from_config(cls, config):
        base_optimizer = keras.optimizers.deserialize(config.pop("base_optimizer"))
        return cls(base_optimizer, **config)