import numpy as np
from keras.src import backend
from keras.src import ops
from keras.src import optimizers
from keras.src import saving
from keras.src.backend import distribution_lib

class TensorParallelOptimizer(optimizers.Optimizer):
    """Consolidated Keras Optimizer for tensor-parallel distributed training."""

    def __init__(
        self,
        base_optimizer,
        device_count,
        shard_optimizer_states=True,
        tensor_parallel_config=None,
        name=None,
        **kwargs,
    ):
        # 1. Essential: Call super().__init__ FIRST
        # We use a placeholder LR of 0.0 because we will sync it 
        # with the base_optimizer immediately after.
        super().__init__(learning_rate=0.0, name=name, **kwargs)

        # 2. Initialize Base Optimizer
        if isinstance(base_optimizer, str):
            self.base_optimizer = optimizers.get(base_optimizer)
        else:
            self.base_optimizer = base_optimizer

        # 3. Sync Learning Rate from base to self
        lr = self.base_optimizer.learning_rate
        if callable(lr):
            self.learning_rate = float(ops.convert_to_numpy(lr(0)))
        else:
            self.learning_rate = float(ops.convert_to_numpy(lr))

        # 4. Distributed Training Config
        self.device_count = device_count
        self.shard_optimizer_states = shard_optimizer_states
        self.tensor_parallel_config = tensor_parallel_config
        
        # 5. Internal State Management
        self._sharded_states = {}
        self._state_var_to_param = {}
        self._var_to_slot_name = {}
        self._model_variables = None

    def build(self, variables):
        """Initializes the optimizer and shards its state variables."""
        if self.built:
            return
        
        self._model_variables = variables
        self.base_optimizer.build(variables)

        # Dry run to initialize optimizer slots (moments, velocity, etc.)
        if variables:
            # Standardize dtypes for safety
            zero_grads = [
                ops.zeros(v.shape, dtype=backend.standardize_dtype(v.dtype)) 
                for v in variables
            ]
            self.base_optimizer.apply_gradients(zip(zero_grads, variables))
        
        # Initialize sharding map
        if self.shard_optimizer_states:
            self._initialize_sharded_states()
            
        super().build(variables)

    def _initialize_sharded_states(self):
        """Maps optimizer slots to model parameters and partitions them."""
        self._sharded_states = {}
        for state_var in self.base_optimizer.variables:
            # Handle the 'iterations' global variable
            if "iterations" in state_var.path or state_var is self.base_optimizer.iterations:
                self._sharded_states["iterations"] = self._partition_state(state_var, 0)
                continue

            # Map state variables to model parameters
            for model_var in self._model_variables:
                if model_var.path in state_var.path:
                    # Extract the slot name (e.g., 'm' or 'v' for Adam)
                    suffix = state_var.path.split(model_var.path)[-1]
                    slot_name = suffix.strip("/")
                    
                    self._state_var_to_param[state_var.path] = model_var
                    self._var_to_slot_name[state_var.path] = slot_name
                    
                    dim = self._get_sharding_dim(model_var)
                    partitioned = self._partition_state(state_var, dim)
                    self._sharded_states.setdefault(slot_name, {})[model_var.path] = partitioned
                    break

    def _partition_state(self, state_variable, dim):
        """Splits a state array into chunks for each device."""
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

    def apply_gradients(self, grads_and_vars, **kwargs):
        """Coordinates gradient synchronization and application."""
        # Determine if we have nested lists (sharded mode)
        is_sharded = (
            isinstance(grads_and_vars, list) and 
            len(grads_and_vars) > 0 and 
            isinstance(grads_and_vars[0], list)
        )
        
        if not is_sharded:
            return self.base_optimizer.apply_gradients(grads_and_vars, **kwargs)

        shard_models = kwargs.get("shard_models")
        if not shard_models:
            raise ValueError("`shard_models` is required when applying sharded gradients.")

        # 1. All-Reduce non-parallel gradients
        synced_grads_and_vars = self._synchronize_gradients(grads_and_vars)

        # 2. Update each shard locally
        for i in range(self.device_count):
            shard_opt = shard_models[i].optimizer.base_optimizer
            
            # Shard -> Local Optimizer
            self._transfer_state(shard_opt, shard_idx=i, direction="to_local")
            
            # Local Update
            shard_opt.apply_gradients(synced_grads_and_vars[i])
            
            # Local Optimizer -> Shard
            self._transfer_state(shard_opt, shard_idx=i, direction="to_global")

    def _transfer_state(self, local_opt, shard_idx, direction="to_local"):
        """Syncs data between sharded numpy storage and local optimizer variables."""
        for var in local_opt.variables:
            # Handle Iterations
            if var is local_opt.iterations:
                if direction == "to_local":
                    var.assign(ops.cast(self._sharded_states["iterations"][shard_idx], var.dtype))
                else:
                    self._sharded_states["iterations"][shard_idx] = ops.convert_to_numpy(var)
                continue

            # Handle Slots
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
        """Cross-device gradient reduction."""
        if not self.tensor_parallel_config:
            return gradients_and_vars

        for i in range(len(gradients_and_vars[0])):
            var = gradients_and_vars[0][i][1]
            # Reduce gradients for variables NOT covered by TP rules
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

    def get_config(self):
        config = super().get_config()
        config.update({
            "base_optimizer": saving.serialize_keras_object(self.base_optimizer),
            "device_count": self.device_count,
            "shard_optimizer_states": self.shard_optimizer_states,
            "tensor_parallel_config": self.tensor_parallel_config,
        })
        return config

    @property
    def variables(self): return self.base_optimizer.variables

    @property
    def iterations(self): return self.base_optimizer.iterations