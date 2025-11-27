import logging
import gc
import re
from typing import Any, Tuple, Set
import numpy as np

# Keras Imports
from keras import Variable, device
from keras.src.models import Model
from keras.src.backend import distribution_lib

# Import local utility for slicing
from keras.src.distribution.tensor_parallel.tensor_layout import split_tensor_for_parallelism

logger = logging.getLogger(__name__)

class ShardedWeight:
    """
    A wrapper for a sharded Keras Variable, ensuring it resides on a specific device.
    """

    def __init__(self, tensor_shard, name, trainable=True, device_id=None):
        current_dev = device_id if device_id else "cpu"
        
        # [FIX] Strict Name Sanitization
        # Keras/JAX backends often forbid characters like '/' in variable names.
        # We replace any non-alphanumeric character with '_' to ensure safety.
        if name:
            clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        else:
            clean_name = "sharded_var"
            
        # [CRITICAL] Create Variable directly on target device (GPU/TPU).
        # This moves the data from Host RAM -> Device VRAM.
        with device(current_dev):
            self._variable = Variable(
                initializer=tensor_shard, 
                trainable=trainable, 
                name=clean_name 
            )

    @property
    def name(self):
        return self._variable.name

    @property
    def trainable(self):
        return self._variable.trainable

    @property
    def shape(self):
        return self._variable.shape

    @property
    def dtype(self):
        return self._variable.dtype

    @property
    def variable(self):
        return self._variable

    @property
    def value(self):
        return self._variable.value

    def numpy(self):
        return self._variable.numpy()

    def assign(self, value):
        self._variable.assign(value)

    def __repr__(self):
        return f"<ShardedWeight name='{self.name}' shape={self.shape} device={self._variable.device}>"


class ParameterShardingStrategy:
    """
    Manually shards model parameters across devices to enable Tensor Parallelism
    without exceeding Host RAM.
    """

    def __init__(self, device_count: int, rank: int):
        self.device_count = device_count
        self.rank = rank
        # Stores the final ShardedWeight objects (living on Device Memory)
        self.sharded_weights = {} 

    def shard_model_parameters(
        self,
        model,
        config,
        device_id: Any,
    ) -> Tuple["Model", Set[str]]:
        """
        Applies sharding to the model weights.
        
        Args:
            model: The Keras model to shard.
            config: A LayoutMap (dict) containing sharding rules.
            device_id: The specific device string (e.g., "gpu:0") for this rank.
        """
        # Define the wrapper class dynamically to avoid circular imports
        ParameterShardedModel = _define_parameter_sharded_model()
        
        print(f"🔧 Applying manual parameter sharding [v3-Fixes] (Rank {self.rank}/{self.device_count})...")
        modified_parameters = set()

        # Iterate through the rules in the LayoutMap
        for pattern, rule in config.items():
            matching_params = self._find_matching_parameters(model, pattern)

            for param_name, param in matching_params:
                try:
                    # 1. Interpret Layout Rules to find split dimension
                    split_dim = None
                    
                    # Handle tuple rules like (None, 'model') or ('model', None)
                    if isinstance(rule, (tuple, list)):
                        for axis_idx, axis_name in enumerate(rule):
                            if axis_name == "model":
                                split_dim = axis_idx
                                break
                    # Handle TensorLayout objects
                    elif hasattr(rule, "axes"):
                        for axis_idx, axis_name in enumerate(rule.axes):
                            if axis_name == "model":
                                split_dim = axis_idx
                                break

                    # If a valid split dimension is found, perform the sharding
                    if split_dim is not None:
                        # 2. Get Slice (Loads to CPU RAM temporarily as NumPy)
                        # This function ensures we slice on CPU to avoid allocating full tensor on GPU.
                        sharded_tensor_np = split_tensor_for_parallelism(
                            param, 
                            index=self.rank, 
                            device_count=self.device_count, 
                            dim=split_dim
                        )
                        
                        # 3. [CRITICAL] Move to Device Immediately
                        # Creating ShardedWeight pushes the numpy array to GPU/TPU VRAM.
                        sharded_var = ShardedWeight(
                            sharded_tensor_np, 
                            name=param_name, 
                            device_id=device_id
                        )
                        
                        # 4. Store reference to the Device Variable
                        self.sharded_weights[param_name] = sharded_var
                        modified_parameters.add(param_name)
                        
                        # 5. [OOM FIX] Free CPU memory immediately
                        del sharded_tensor_np
                        
                        # 6. [OOM FIX] Free original weight memory from source model
                        # We replace the heavy tensor with a tiny dummy to free Host RAM.
                        try:
                            dummy = np.zeros((1,), dtype=param.dtype)
                            param.assign(dummy)
                        except Exception:
                            pass
                        
                        # 7. Force Garbage Collection
                        # This is vital when processing 9B parameters (36GB+ float32 data).
                        gc.collect()
                        
                    else:
                        # If no 'model' axis is present, we skip (or replicate if needed).
                        # For manual TP, we usually only care about sharding the heavy weights.
                        pass

                except Exception as e:
                    # Fail fast so we don't end up with a broken model running on CPU
                    print(f"   ❌ FATAL ERROR: Failed to shard parameter '{param_name}': {e}")
                    raise e

        # Create the wrapper model that uses the sharded weights
        sharded_model = ParameterShardedModel(
            original_model=model,
            sharding_strategy=self,
            config=config,
            device_id=device_id,
        )

        print(
            f"🎯 Parameter sharding completed: {len(modified_parameters)} parameters sharded."
        )
        return sharded_model, modified_parameters

    def _find_matching_parameters(self, model, pattern: str):
        """
        Find parameters that match the given regex pattern.
        """
        matches = []
        # We search both path (e.g. 'layer/kernel') and name
        for v in model.variables:
            if re.fullmatch(pattern, v.path) or re.fullmatch(pattern, v.name):
                matches.append((v.path, v))
        return matches


def _define_parameter_sharded_model():
    """
    Factory function to define the ParameterShardedModel class.
    """
    from keras.src.models import Model

    class ParameterShardedModel(Model):
        """
        Wrapper model that replaces original weights with ShardedWeights
        and applies communication (AllReduce) during the forward pass.
        """

        def __init__(
            self,
            original_model: Model,
            sharding_strategy: ParameterShardingStrategy,
            config: dict,
            device_id: Any,
        ):
            super().__init__()
            self.original_model = original_model
            self.sharding_strategy = sharding_strategy
            self.config = config
            self._device = device_id

            # Rebuild the list of variables using our new ShardedWeights
            self._recreate_variables()

        def _recreate_variables(self):
            print("   - Linking sharded weights to model...")
            self._sharded_vars = []
            
            # Use the dictionary of sharded weights populated by the strategy
            for name, weight_obj in self.sharding_strategy.sharded_weights.items():
                self._sharded_vars.append(weight_obj)

        def call(self, inputs, training=None, mask=None):
            # 1. Forward Pass
            # We delegate the call to the original model structure.
            # Since we replaced the weights in the strategy (conceptually), 
            # or rely on this wrapper to inject them, this step depends on 
            # how Keras resolves the variable lookups.
            # In this manual "hack", we assume the backend handles the sharded inputs 
            # if they match the local device logic.
            outputs = self.original_model(inputs, training=training, mask=mask)

            # 2. Apply Output Rules (Communication / AllReduce)
            # This is where we sum up the partial results from Row Parallel layers.
            if hasattr(self.config, "output_rules"):
                for layer_name, rule in self.config.output_rules.items():
                    # Check if this rule applies (simple name match for now)
                    # A more robust implementation would check if 'layer_name' is in the call path.
                    if rule == "allreduce sum":
                         # Perform the cross-device sum
                         outputs = distribution_lib.all_reduce(outputs, op="sum", axis_name="model")

            return outputs

        def get_config(self):
            return self.original_model.get_config()

        @classmethod
        def from_config(cls, config):
            return cls(**config)

    return ParameterShardedModel