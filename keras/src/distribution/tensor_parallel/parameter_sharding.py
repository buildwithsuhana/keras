import logging
import gc
import re
from typing import Any, Tuple, Set
import numpy as np
from keras import Variable, device, ops
from keras.src import layers
from keras.src.models import Model
from keras.src.backend import distribution_lib

# Import our custom logic
from keras.src.distribution.tensor_parallel.tensor_layout import split_tensor_for_parallelism

logger = logging.getLogger(__name__)

class ShardedWeight:
    """Wrapper for a sharded variable on a specific device."""
    def __init__(self, tensor_shard, name, trainable=True, device_id=None):
        current_dev = device_id if device_id else "cpu"
        with device(current_dev):
            self._variable = Variable(
                initializer=tensor_shard, 
                trainable=trainable, 
                name=name
            )

    @property
    def value(self): return self._variable.value
    @property
    def numpy(self): return self._variable.numpy
    @property
    def shape(self): return self._variable.shape
    @property
    def dtype(self): return self._variable.dtype
    def assign(self, value): self._variable.assign(value)


class ParameterShardingStrategy:
    def __init__(self, device_count: int, rank: int):
        self.device_count = device_count
        self.rank = rank
        self.sharded_weights = {}

    def shard_model_parameters(self, model, config, device_id: Any) -> Tuple["Model", Set[str]]:
        """
        Manually shards model parameters based on the LayoutMap config.
        """
        ParameterShardedModel = _define_parameter_sharded_model()
        
        print(f"🔧 Applying manual parameter sharding (Rank {self.rank}/{self.device_count})...")
        modified_parameters = set()

        # config is our LayoutMap (dict-like)
        for pattern, rule in config.items():
            matching_params = self._find_matching_parameters(model, pattern)

            for param_name, param in matching_params:
                try:
                    # --- [FIX] Interpret Layout Rules ---
                    split_dim = None
                    
                    # Heuristic: Find which axis has 'model'
                    if isinstance(rule, (tuple, list)):
                        for axis_idx, axis_name in enumerate(rule):
                            if axis_name == "model":
                                split_dim = axis_idx
                                break
                    
                    # If rule is a class (TensorLayout), try to read its axes
                    elif hasattr(rule, "axes"):
                        for axis_idx, axis_name in enumerate(rule.axes):
                            if axis_name == "model":
                                split_dim = axis_idx
                                break

                    # If valid split dimension found, perform the split
                    if split_dim is not None:
                        sharded_tensor_np = split_tensor_for_parallelism(
                            param, 
                            index=self.rank, 
                            device_count=self.device_count, 
                            dim=split_dim
                        )
                        
                        self.sharded_weights[param_name] = sharded_tensor_np
                        modified_parameters.add(param_name)
                        
                        # [OOM FIX] Free original weight memory aggressively
                        try:
                            # Replace with tiny dummy to free Host RAM
                            dummy = np.zeros((1,), dtype=param.dtype)
                            param.assign(dummy)
                        except:
                            pass
                        gc.collect()
                        
                    else:
                        pass # Replicate or ignore

                except Exception as e:
                    print(f"   ❌ Failed to shard {param_name}: {e}")
                    continue

        sharded_model = ParameterShardedModel(
            original_model=model,
            sharding_strategy=self,
            config=config,
            device_id=device_id,
        )
        
        return sharded_model, modified_parameters

    def _find_matching_parameters(self, model, pattern):
        matches = []
        for v in model.variables:
            if re.fullmatch(pattern, v.path) or re.fullmatch(pattern, v.name):
                matches.append((v.path, v))
        return matches


def _define_parameter_sharded_model():
    from keras.src.models import Model
    
    class ParameterShardedModel(Model):
        def __init__(self, original_model, sharding_strategy, config, device_id):
            super().__init__()
            self.original_model = original_model
            self.sharding_strategy = sharding_strategy
            self.config = config
            self.device_id = device_id
            
            self._recreate_variables()

        def _recreate_variables(self):
            # Map sharded weights to new variables on the target device
            self._sharded_vars = []
            for name, shard_np in self.sharding_strategy.sharded_weights.items():
                w = ShardedWeight(shard_np, name, device_id=self.device_id)
                self._sharded_vars.append(w)
                
        def call(self, inputs, training=None, mask=None):
            # 1. Forward pass
            outputs = self.original_model(inputs, training=training, mask=mask)
            
            # 2. Apply Output Rules (AllReduce)
            if hasattr(self.config, "output_rules"):
                for layer_name, rule in self.config.output_rules.items():
                    if rule == "allreduce sum":
                         outputs = distribution_lib.all_reduce(outputs, op="sum", axis_name="model")
            
            return outputs

    return ParameterShardedModel