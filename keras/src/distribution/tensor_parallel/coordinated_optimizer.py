import numpy as np
from keras.src import backend
from keras.src import ops
from keras.src import optimizers
from keras.src import saving
from keras.src.backend import distribution_lib

class TensorParallelOptimizer(optimizers.Optimizer):
    """Consolidated Keras Optimizer for tensor-parallel distributed training.
    
    This class handles gradient synchronization, state sharding, and weight 
    updates across multiple devices in a single class.
    """

    def __init__(
        self,
        base_optimizer,
        device_count,
        shard_optimizer_states=True,
        tensor_parallel_config=None,
        name=None,
        **kwargs,
    ):
        # 1. Initialize Base Optimizer
        if isinstance(base_optimizer, str):
            self.base_optimizer = optimizers.get(base_optimizer)
        else:
            self.base_optimizer = base_optimizer

        # 2. Extract LR and Setup Keras Optimizer properties
        lr = self.base_optimizer.learning_rate
        lr_val = float(ops.convert_to_numpy(lr(0))) if callable(lr) else float(ops.convert_to_numpy(lr))
        
        name = name or f"TensorParallel_{self.base_optimizer.name}"
        kwargs.pop("learning_rate", None)
        super().__init__(learning_rate=lr_val, name=name, **kwargs)

        # 3. Distributed Training Config
        self.device_count = device_count
        self.shard_optimizer_states = shard_optimizer_states
        self.tensor_parallel_config = tensor_parallel_config
        
        # 4. Internal State Management
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
            zero_grads = [ops.zeros(v.shape, dtype=v.dtype) for v in variables]
            self.base_optimizer.apply_gradients(zip(zero_grads, variables))
        
        # Initialize sharding map
        if self.shard_optimizer_states:
            self._initialize_sharded_states()
            
        super().build(variables)

    # --- Sharding Logic ---

    def _initialize_sharded_states(self):
        """Maps optimizer slots to model parameters and partitions them."""
        self._sharded_states = {}
        for state_var in self.base_optimizer.variables:
            if state_var is self.base_optimizer.iterations:
                self._sharded_states["iterations"] = self._partition_state(state_var, 0)
                continue

            # Identify which model parameter this state variable belongs to
            for model_var in self._model_variables:
                if model_var.path in state_var.path:
                    slot_name = state_var.path.split(model_var.path)[-1].strip("/")
                    self._state_var_to_param[state_var.path] = model_var
                    self._var_to_slot_name[state_var.path] = slot_name
                    
                    # Determine sharding dimension
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
        """Retrieves sharding dimension from config if available."""
        if not self.tensor_parallel_config:
            return 0
        rule = self.tensor_parallel_config.state_rules.get(id(param))
        if rule:
            return rule.keywords.get("dim", 0) if hasattr(rule, "keywords") else getattr(rule, "dim", 0)
        return 0

    # --- Gradient Application ---

    def apply_gradients(self, grads_and_vars, **kwargs):
        """Main entry point for updates."""
        # Check if we are receiving a list of lists (sharded gradients)
        is_sharded = isinstance(grads_and_vars, list) and isinstance(grads_and_vars[0], list)
        
        if not is_sharded:
            return self.base_optimizer.apply_gradients(grads_and_vars, **kwargs)

        shard_models = kwargs.get("shard_models")
        if not shard_models:
            raise ValueError("`shard_models` is required for sharded gradient updates.")

        # 1. Sync gradients across devices (All-Reduce)
        synced_grads_and_vars = self._synchronize_gradients(grads_and_vars)

        # 2. Apply updates shard-by-shard
        for i in range(self.device_count):
            shard_opt = shard_models[i].optimizer.base_optimizer
            
            # Load local shard of optimizer state
            self._transfer_state(shard_opt, shard_idx=i, direction="to_local")
            
            # Perform update
            shard_opt.apply_gradients(synced_grads_and_vars[i])
            
            # Save updated state back to global sharded storage
            self._transfer_state(shard_opt, shard_idx=i, direction="to_global")

    def _transfer_state(self, local_opt, shard_idx, direction="to_local"):
        """Moves data between the local optimizer instance and global sharded state."""
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
        """Performs All-Reduce logic based on TP configuration."""
        if not self.tensor_parallel_config:
            return gradients_and_vars

        # Simplify: Perform mean All-Reduce for non-tensor-parallel weights
        for i in range(len(gradients_and_vars[0])):
            var = gradients_and_vars[0][i][1]
            if var.path not in self.tensor_parallel_config.state_rules:
                grads = [shard[i][0] for shard in gradients_and_vars if shard[i][0] is not None]
                if grads:
                    # Use backend communication if available
                    if distribution_lib.get_device_count() > 1:
                        reduced = distribution_lib.all_reduce(grads[0], op="mean", axis_name="model")
                    else:
                        reduced = ops.mean(ops.stack(grads), axis=0)
                    
                    for shard_idx in range(self.device_count):
                        gradients_and_vars[shard_idx][i] = (reduced, var)
        return gradients_and_vars

    # --- Keras Boilerplate ---

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
    def learning_rate(self): return self.base_optimizer.learning_rate

    @learning_rate.setter
    def learning_rate(self, value): self.base_optimizer.learning_rate = value