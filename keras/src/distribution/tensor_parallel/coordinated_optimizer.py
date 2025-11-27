import re
from typing import Any, List, Tuple

import numpy as np
from keras.src import saving

from keras.src import ops
from keras.src import optimizers

from keras.src.backend import distribution_lib


class CoordinatedOptimizer:
    """Manages an optimizer's state for distributed training.
    
    This class handles the synchronization of gradients for Replicated variables
    (variables that exist on all shards and must remain identical) and delegates
    the actual update step to the base optimizer.
    """

    def __init__(
        self,
        base_optimizer: optimizers.Optimizer,
        device_count: int,
        shard_optimizer_states: bool = False, # Changed default to False for OOM safety
        tensor_parallel_config=None,
    ):
        self.base_optimizer = base_optimizer
        self.device_count = device_count
        self.shard_optimizer_states = shard_optimizer_states
        self.tensor_parallel_config = tensor_parallel_config
        
        # Mapping injected by TensorParallelKeras
        # Structure: { 'logical_path': [var_shard0, var_shard1, ...] }
        self._shard_var_map = {}
        self._shard_models = []

    def enable_optimizer_state_sharding(self, variables: list):
        """
        Placeholder for enabling advanced state sharding (ZeRO).
        Currently disabled/simplified to prevent OOM on host.
        """
        self._variables = variables
        # We skip the heavy numpy conversion logic here to save memory.
        pass

    def apply_gradients(self, grads_and_vars: List[Tuple[Any, Any]]):
        """Coordinates gradient synchronization and application."""
        
        # 1. Synchronize gradients for Replicated variables
        # (Variables that are NOT sharded by TP rules must be averaged across devices)
        synced_grads_and_vars = self._synchronize_gradients(grads_and_vars)

        # 2. Apply gradients using the base optimizer
        # The base optimizer maintains separate states (m, v) for every variable instance,
        # so it naturally handles the distributed variables correctly.
        self.base_optimizer.apply_gradients(synced_grads_and_vars)

    def _synchronize_gradients(
        self, gradients_and_vars: List[Tuple[Any, Any]]
    ) -> List[Tuple[Any, Any]]:
        """Synchronizes gradients for replicated variables."""
        if not self._shard_var_map:
            return gradients_and_vars

        # Map variable object ID to index in the grads list for fast lookup
        var_to_idx = {id(v): i for i, (g, v) in enumerate(gradients_and_vars)}
        
        for logical_path, shard_vars in self._shard_var_map.items():
            # If a variable exists on < 2 shards, no sync needed.
            if len(shard_vars) < 2:
                continue

            # Determine if this variable is Sharded or Replicated.
            is_sharded = False
            if self.tensor_parallel_config:
                for pattern in self.tensor_parallel_config.state_rules.keys():
                    if re.search(pattern, logical_path):
                        is_sharded = True
                        break
            
            # If TP rules sharded this variable (e.g. Dense Kernel), 
            # the shards are DISTINCT parts. Do NOT average them.
            if is_sharded:
                continue
                
            # If we are here, the variable is Replicated (e.g. Bias, LayerNorm).
            # We must average gradients across all shards to keep them in sync.
            grads_to_average = []
            indices_to_update = []
            
            valid_sync = True
            for v in shard_vars:
                idx = var_to_idx.get(id(v))
                if idx is not None:
                    g = gradients_and_vars[idx][0]
                    if g is not None:
                        grads_to_average.append(g)
                        indices_to_update.append(idx)
                    else:
                        # One replica has no grad? Skip sync (unlikely in valid training)
                        valid_sync = False
                else:
                    # Variable not trainable?
                    valid_sync = False

            if valid_sync and len(grads_to_average) > 1:
                # Stack and Mean
                # Note: This runs on the device where the grads are located 
                # (or triggers transfer if devices differ, handled by backend).
                try:
                    grads_t = [ops.convert_to_tensor(g) for g in grads_to_average]
                    stacked = ops.stack(grads_t)
                    avg_grad = ops.mean(stacked, axis=0)
                    
                    # Update the gradients list in-place
                    for idx in indices_to_update:
                        original_var = gradients_and_vars[idx][1]
                        gradients_and_vars[idx] = (avg_grad, original_var)
                except Exception as e:
                    # Fallback or log warning if shapes mismatch (should not happen for replicated vars)
                    print(f"Warning: Failed to sync gradients for {logical_path}: {e}")

        return gradients_and_vars

    def get_weights(self) -> list[np.ndarray]:
        """Returns the weights of the base optimizer."""
        return self.base_optimizer.get_weights()

    def set_weights(self, weights: list[np.ndarray]):
        """Sets the weights of the base optimizer."""
        self.base_optimizer.set_weights(weights)


class TensorParallelOptimizer(optimizers.Optimizer):
    """A Keras Optimizer wrapper for tensor-parallel distributed training."""

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

        # --- FIX START: robust learning_rate handling ---
        learning_rate = base_optimizer_instance.learning_rate
        
        # If the base optimizer has already converted a float LR to a Variable (common in Keras 3),
        # we must extract the float value because the Optimizer.__init__ validation 
        # rejects Variable objects (it expects float, callable, or schedule).
        if not callable(learning_rate):
            try:
                # ops.convert_to_numpy works for JAX/TF/Torch backends
                learning_rate = float(ops.convert_to_numpy(learning_rate))
            except Exception:
                # If extraction fails (e.g. symbolic tensor), fallback to safe default.
                # This is just for initialization; the base optimizer holds the real state.
                learning_rate = 0.001
        # --- FIX END ---

        super().__init__(
            learning_rate=learning_rate,
            name=f"TensorParallel_{base_optimizer_instance.name}",
        )

        self.base_optimizer = base_optimizer_instance
        self.device_count = device_count
        self.coordinated_optimizer = CoordinatedOptimizer(
            self.base_optimizer,
            device_count,
            tensor_parallel_config=tensor_parallel_config,
        )
        
        # These will be populated by TensorParallelKeras.compile()
        self._shard_models = []
        self._shard_var_map = {}

    def apply_gradients(self, grads_and_vars: list, **kwargs):
        """Applies gradients to the model variables."""
        # Ensure the coordinator has the latest map
        if self._shard_var_map and not self.coordinated_optimizer._shard_var_map:
             self.coordinated_optimizer._shard_var_map = self._shard_var_map

        # Delegate to coordinator
        self.coordinated_optimizer.apply_gradients(grads_and_vars)

    def update_step(self, gradient, variable, *args, **kwargs):
        """Delegates the update step to the base optimizer."""
        if hasattr(self.base_optimizer, "update_step"):
            try:
                return self.base_optimizer.update_step(
                    gradient, variable, *args, **kwargs
                )
            except TypeError:
                return self.base_optimizer.update_step(gradient, variable)

        try:
            return super().update_step(gradient, variable, *args, **kwargs)
        except TypeError:
            return super().update_step(gradient, variable)

    def build(self, variables: list):
        """Builds the optimizer."""
        if self.built:
            return

        self.base_optimizer.build(variables)
        self.coordinated_optimizer.enable_optimizer_state_sharding(variables)
        super().build(variables)

    def get_weights(self) -> list[np.ndarray]:
        return self.coordinated_optimizer.get_weights()

    def set_weights(self, weights: list[np.ndarray]):
        self.coordinated_optimizer.set_weights(weights)

    @property
    def variables(self) -> list:
        return self.base_optimizer.variables

    @property
    def learning_rate(self) -> Any:
        return self.base_optimizer.learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        self.base_optimizer.learning_rate = value

    @property
    def iterations(self):
        return self.base_optimizer.iterations