import logging
import re
import gc
from typing import Any, Tuple, Set, TYPE_CHECKING
import keras
from keras import Variable, device
from keras.src.models import Model

logger = logging.getLogger(__name__)

class ParameterShardingStrategy:
    def __init__(self, device_count: int, rank: int):
        self.device_count = device_count
        self.rank = rank

    def shard_model_parameters(self, replica_model, config, device_id):
        ParameterShardedModel = _define_parameter_sharded_model()
        
        modified = set()
        print(f"   🔧 Sharding parameters for Rank {self.rank} on {device_id}...")

        # Iterate config rules
        for pattern, action in config.state_rules.items():
            if callable(action):
                matches = self._find_matches(replica_model, pattern)
                for layer, attr, param in matches:
                    # 1. Get CPU Value
                    val = param.value
                    
                    # 2. Slice (on CPU)
                    try:
                        sharded_val = action(val, self.rank)
                    except Exception as e:
                        logger.warning(f"Failed to shard {param.name}: {e}")
                        continue
                        
                    # 3. Create GPU Variable
                    with device(device_id):
                        new_var = Variable(
                            initializer=sharded_val,
                            trainable=param.trainable,
                            name=param.name,
                            dtype=param.dtype
                        )
                    
                    # 4. Replace in Layer
                    setattr(layer, attr, new_var)
                    modified.add(param.name)
                    
                    # 5. Clear CPU Temps
                    del sharded_val
        
        return ParameterShardedModel(replica_model, config, device_id), modified

    def _find_matches(self, model, pattern):
        matches = []
        seen = set()
        
        def recurse(obj, prefix=""):
            if id(obj) in seen: return
            seen.add(id(obj))
            
            if isinstance(obj, (keras.layers.Layer, keras.Model)):
                name = obj.name
                full = f"{prefix}.{name}" if prefix else name
                
                # Check attributes
                for attr in dir(obj):
                    if attr.startswith("_"): continue
                    try:
                        val = getattr(obj, attr)
                    except: continue
                    
                    if isinstance(val, Variable):
                        clean_name = val.name.split("/")[-1].split(":")[0]
                        candidate = f"{full}.{clean_name}"
                        if re.fullmatch(pattern, candidate):
                            matches.append((obj, attr, val))
                    
                    if isinstance(val, (keras.layers.Layer, keras.Model)):
                        recurse(val, full)
                    elif isinstance(val, (list, tuple)):
                        for i in val:
                            if isinstance(i, (keras.layers.Layer, keras.Model)):
                                recurse(i, full)
        recurse(model)
        return matches

def _define_parameter_sharded_model():
    from keras.src.backend import distribution_lib
    
    class ParameterShardedModel(Model):
        def __init__(self, model_shard, config, device_id):
            super().__init__(name=model_shard.name)
            self.model_shard = model_shard
            self.config = config
            self._device = device_id
            
        @property
        def device(self): return self._device
        @property
        def weights(self): return self.model_shard.weights
        @property
        def trainable_weights(self): return self.model_shard.trainable_weights
        @property
        def non_trainable_weights(self): return self.model_shard.non_trainable_weights

        def call(self, inputs, training=None, mask=None):
            out = self.model_shard(inputs, training=training, mask=mask)
            
            # Basic communication rule application if needed
            # (Assume most layers handle it internally or via config rules)
            layer_name = self.model_shard.layers[-1].name
            rule = self.config.output_rules.get(layer_name)
            if rule:
                rule_str = rule.get(0, "")
                if "sum" in rule_str or "allreduce" in rule_str:
                    out = distribution_lib.all_reduce(out, op="sum", axis_name="model")
                elif "gather" in rule_str:
                    out = distribution_lib.all_gather(out, axis=-1, axis_name="model")
            
            return out
            
    return ParameterShardedModel

def make_parameter_sharded_model(replica, config, rank, device_count, device_id):
    strategy = ParameterShardingStrategy(device_count, rank)
    return strategy.shard_model_parameters(replica, config, device_id)