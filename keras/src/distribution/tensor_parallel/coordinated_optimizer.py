import numpy as np
from keras.src import backend
from keras.src import ops
from keras.src import optimizers
from keras.src import saving
from keras.src.backend import distribution_lib

class TensorParallelOptimizer(optimizers.Optimizer):
    def __init__(
        self,
        base_optimizer,
        device_count,
        shard_optimizer_states=True,
        tensor_parallel_config=None,
        name=None,
        **kwargs,
    ):
        super().__init__(learning_rate=0.0, name=name, **kwargs)

        if isinstance(base_optimizer, str):
            self.base_optimizer = optimizers.get(base_optimizer)
        else:
            self.base_optimizer = base_optimizer

        # Sync Learning Rate
        lr = self.base_optimizer.learning_rate
        if callable(lr):
            self.learning_rate = float(ops.convert_to_numpy(lr(0)))
        else:
            self.learning_rate = float(ops.convert_to_numpy(lr))

        self.device_count = device_count
        self.shard_optimizer_states = shard_optimizer_states
        self.tensor_parallel_config = tensor_parallel_config
        
        self._sharded_states = {}
        self._state_var_to_param = {}
        self._var_to_slot_name = {}
        self._model_variables = None

    def build(self, variables):
        if self.built:
            return
        
        self._model_variables = variables
        self.base_optimizer.build(variables)

        if variables:
            zero_grads = [
                ops.zeros(v.shape, dtype=backend.standardize_dtype(v.dtype)) 
                for v in variables
            ]
            self.base_optimizer.apply_gradients(zip(zero_grads, variables))
        
        if self.shard_optimizer_states:
            self._initialize_sharded_states()
            
        super().build(variables)

    def update_step(self, gradient, variable, learning_rate=None):
        """Required for Keras's fit() and standard apply_gradients path."""
        return self.base_optimizer.update_step(
            gradient, variable, learning_rate=learning_rate
        )

    def apply_gradients(self, grads_and_vars, **kwargs):
        is_sharded = (
            isinstance(grads_and_vars, list) and 
            len(grads_and_vars) > 0 and 
            isinstance(grads_and_vars[0], list)
        )
        
        if not is_sharded:
            # Fall back to standard Keras update logic
            return super().apply_gradients(grads_and_vars, **kwargs)

        shard_models = kwargs.get("shard_models")
        if not shard_models:
            raise ValueError("`shard_models` is required for sharded gradients.")

        synced_grads_and_vars = self._synchronize_gradients(grads_and_vars)

        for i in range(self.device_count):
            shard_opt = shard_models[i].optimizer.base_optimizer
            self._transfer_state(shard_opt, shard_idx=i, direction="to_local")
            shard_opt.apply_gradients(synced_grads_and_vars[i])
            self._transfer_state(shard_opt, shard_idx=i, direction="to_global")

    # --- Helper methods remain the same as previous step ---
    def _initialize_sharded_states(self):
        self._sharded_states = {}
        for state_var in self.base_optimizer.variables:
            if "iterations" in state_var.path or state_var is self.base_optimizer.iterations:
                self._sharded_states["iterations"] = self._partition_state(state_var, 0)
                continue
            for model_var in self._model_variables:
                if model_var.path in state_var.path:
                    suffix = state_var.path.split(model_var.path)[-1]
                    slot_name = suffix.strip("/")
                    self._state_var_to_param[state_var.path] = model_var
                    self._var_to_slot_name[state_var.path] = slot_name
                    dim = self._get_sharding_dim(model_var)
                    partitioned = self._partition_state(state_var, dim)
                    self._sharded_states.setdefault(slot_name, {})[model_var.path] = partitioned
                    break

    def _partition_state(self, state_variable, dim):
        arr = ops.convert_to_numpy(state_variable)
        if arr.ndim > dim and arr.shape[dim] >= self.device_count:
            return np.array_split(arr, self.device_count, axis=dim)
        return [np.copy(arr) for _ in range(self.device_count)]

    def _get_sharding_dim(self, param):
        if not self.tensor_parallel_config:
            return 0
        rule = self.tensor_parallel_config.state_rules.get(id(param))
        if rule:
            if hasattr(rule, "keywords") and "dim" in rule.keywords:
                return rule.keywords["dim"]
            return getattr(rule, "dim", 0)
        return 0

    def _transfer_state(self, local_opt, shard_idx, direction="to_local"):
        for var in local_opt.variables:
            if var is local_opt.iterations:
                if direction == "to_local":
                    var.assign(ops.cast(self._sharded_states["iterations"][shard_idx], var.dtype))
                else:
                    self._sharded_states["iterations"][shard_idx] = ops.convert_to_numpy(var)
                continue
            param = self._state_var_to_param.get(var.path)
            slot = self._var_to_slot_name.get(var.path)
            if param and slot in self._sharded_states and param.path in self._sharded_states[slot]:
                if direction == "to_local":
                    val = self._sharded_states[slot][param.path][shard_idx]
                    if var.shape == val.shape:
                        var.assign(ops.cast(val, var.dtype))
                else:
                    self._sharded_states[slot][param.path][shard_idx] = ops.convert_to_numpy(var)

    def _synchronize_gradients(self, gradients_and_vars):
        if not self.tensor_parallel_config:
            return gradients_and_vars
        for i in range(len(gradients_and_vars[0])):
            var = gradients_and_vars[0][i][1]
            if var.path not in self.tensor_parallel_config.state_rules:
                grads = [shard[i][0] for shard in gradients_and_vars if shard[i][0] is not None]
                if grads:
                    if distribution_lib.get_device_count() > 1:
                        reduced = distribution_lib.all_reduce(grads[0], op="mean", axis_name="model")
                    else:
                        reduced = ops.mean(ops.stack(grads), axis=0)
                    for shard_idx in range(self.device_count):
                        gradients_and_vars[shard_idx][i] = (reduced, var)
        return gradients_and_vars

    @property
    def variables(self): return self.base_optimizer.variables

    @property
    def iterations(self): return self.base_optimizer.iterations