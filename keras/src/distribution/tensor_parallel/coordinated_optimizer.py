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

    def apply_gradients(
        self, gradients_and_vars: list[list[tuple]], shard_models: list
    ):
        # Synchronize gradients across shards for parameters that require reduction
        synchronized_gradients = self._synchronize_gradients(gradients_and_vars)

        for shard_idx in range(self.device_count):
            shard_opt = shard_models[shard_idx].optimizer
            
            # Unwrap potential wrappers to get to the actual optimizer apply_gradients
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

        # DEBUG: Find patterns that require gradient reduction (e.g. Column Parallel)
        # In TP, 'up_projection' and 'qkv' typically require reduction of sharded inputs.
        # Since your autoconfig uses generic split_rules, we look for matches in state_rules.
        rules = self.tensor_parallel_config.state_rules
        
        num_weights = len(gradients_and_vars[0])
        for i in range(num_weights):
            # Check the variable name from the first shard
            variable = gradients_and_vars[0][i][1]
            var_path = getattr(variable, "path", getattr(variable, "name", ""))
            
            # Identify if this variable was sharded using a rule from AutoConfig
            is_sharded = any(re.search(pattern, var_path) for pattern in rules.keys())
            
            if is_sharded:
                grads_to_reduce = [
                    g_and_v[i][0]
                    for g_and_v in gradients_and_vars
                    if g_and_v[i][0] is not None
                ]
                
                if grads_to_reduce:
                    # FIX: Perform reduction on the device of the first gradient to avoid CPU RAM spikes
                    target_device = getattr(grads_to_reduce[0], "device", "gpu:0")
                    
                    with keras.device(target_device):
                        # Start iterative accumulation
                        total_grad = ops.convert_to_tensor(grads_to_reduce[0])
                        for g in grads_to_reduce[1:]:
                            total_grad = ops.add(total_grad, ops.convert_to_tensor(g))
                        
                        mean_grad = ops.divide(total_grad, float(len(grads_to_reduce)))
                    
                    # Distribute the reduced gradient back to all shards for this specific variable
                    for shard_idx in range(self.device_count):
                        if gradients_and_vars[shard_idx][i][0] is not None:
                            gradients_and_vars[shard_idx][i] = (mean_grad, gradients_and_vars[shard_idx][i][1])
                    
                    # print(f"   [DEBUG] Synchronized gradients for: {var_path} on {target_device}")
        
        return gradients_and_vars

    def get_weights(self) -> list[np.ndarray]: return []
    def set_weights(self, weights: list[np.ndarray]): pass
    def enable_optimizer_state_sharding(self, variables: list): pass


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
            # Handle float or LearningRateSchedule
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
        # Identify if we are receiving a list of lists (sharded gradients)
        is_sharded_grads = (
            isinstance(grads_and_vars, list)
            and grads_and_vars
            and isinstance(grads_and_vars[0], list)
        )
        
        if is_sharded_grads:
            shard_models = kwargs.get("shard_models")
            if not shard_models:
                 # Fallback to internal model shards if provided during compilation
                 shard_models = getattr(self, "_shard_models", [])
            
            if shard_models:
                self.coordinated_optimizer.apply_gradients(grads_and_vars, shard_models)
            else:
                print("[WARNING] No shard_models found in apply_gradients. Falling back to base optimizer.")
                self.base_optimizer.apply_gradients(grads_and_vars[0])
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