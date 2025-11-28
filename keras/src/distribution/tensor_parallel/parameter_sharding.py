import logging
import re
import gc
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

import numpy as np
import keras
from keras.src.backend import distribution_lib
from keras.src import ops
from keras.src import layers
from keras import Variable, device 

from keras.src.distribution.tensor_parallel.tensor_layout import LayoutMap

if TYPE_CHECKING:
    from keras.src.models import Model

logger = logging.getLogger(__name__)

class ParameterShardingStrategy:
    def __init__(self, device_count: int, rank: int):
        self.device_count = device_count
        self.rank = rank

    def shard_model_parameters(
        self,
        replica_model: "Model",
        config: LayoutMap,
        device_id: Any,
    ) -> Tuple["Model", Set[str]]:
        
        ParameterShardedModel = _define_parameter_sharded_model()
        print(f"🔧 Applying parameter-level sharding to replica for Rank {self.rank}")
        
        modified_parameters = set()

        # Iterate Config
        for pattern, action in config.state_rules.items():
            if callable(action):
                matching_params = self._find_matching_layers_and_attrs(replica_model, pattern)

                for layer, attr_name, param in matching_params:
                    # Get value from CURRENT replica (CPU)
                    original_value = param.value 
                    
                    # Slice it
                    try:
                        sharded_value = action(original_value, self.rank)
                    except Exception as e:
                        logger.error(f"Error slicing {param.name}: {e}")
                        continue

                    # Create Variable on GPU
                    with device(device_id):
                        new_sharded_var = Variable(
                            initializer=sharded_value,
                            trainable=param.trainable,
                            name=param.name,
                            dtype=param.dtype
                        )
                    
                    # Replace in Replica
                    setattr(layer, attr_name, new_sharded_var)
                    modified_parameters.add(param.name)
                    
                    # Free CPU memory immediately
                    del sharded_value
                    # Note: We can't delete 'original_value' easily as it's bound to the Keras Variable
                    # But Python GC might handle it if we are lucky.

        sharded_model_wrapper = ParameterShardedModel(
            model_shard=replica_model,
            sharding_strategy=self,
            config=config,
            device_id=device_id,
        )
        return sharded_model_wrapper, modified_parameters

    def _find_matching_layers_and_attrs(self, model, pattern: str) -> List[Tuple[layers.Layer, str, Any]]:
        matches = []
        processed_objs = set()

        def search_recursive(obj, prefix=""):
            if id(obj) in processed_objs: return
            processed_objs.add(id(obj))

            if isinstance(obj, (layers.Layer, keras.Model)):
                name = obj.name
                full_name = f"{prefix}.{name}" if prefix else name
                
                for attr_name in dir(obj):
                    if attr_name.startswith("_"): continue
                    try:
                        val = getattr(obj, attr_name)
                    except: continue
                        
                    if isinstance(val, Variable):
                        cleaned = val.name.split("/")[-1].split(":")[0]
                        candidate_name = f"{full_name}.{cleaned}"
                        if re.fullmatch(pattern, candidate_name):
                             matches.append((obj, attr_name, val))
                    
                    if isinstance(val, (layers.Layer, keras.Model)):
                        search_recursive(val, full_name)
                    elif isinstance(val, (list, tuple)):
                        for item in val:
                            if isinstance(item, (layers.Layer, keras.Model)):
                                search_recursive(item, full_name)

        search_recursive(model, prefix="")
        return matches


def _define_parameter_sharded_model():
    from keras.src.models import Model
    class ParameterShardedModel(Model):
        def __init__(self, model_shard: Model, sharding_strategy, config, device_id):
            super().__init__(name=model_shard.name)
            self.model_shard = model_shard
            self.sharding_strategy = sharding_strategy
            self.config = config
            self._device = device_id
            
            # Rebuild if needed for input shape inference
            if not self.model_shard.built and self.model_shard.inputs:
                 pass 

        @property
        def device(self): return self._device
        @property
        def weights(self): return self.model_shard.weights
        @property
        def trainable_weights(self): return self.model_shard.trainable_weights
        @property
        def non_trainable_weights(self): return self.model_shard.non_trainable_weights

        def call(self, inputs, training=None, mask=None):
            # Input reconstruction logic (Same as before)
            tensor_cache = {}
            if isinstance(inputs, dict):
                tensor_cache.update(inputs)
            else:
                flat = ops.convert_to_tensor(inputs)
                # Naive mapping
                pass 

            # Execute Model Shard Layers
            # We must forward call to the underlying model
            # But the underlying model call() might re-create graph logic
            # Since we replaced weights, it should just work.
            return self.model_shard(inputs, training=training, mask=mask)

    return ParameterShardedModel

def make_parameter_sharded_model(
    replica_model: "Model",
    config: LayoutMap,
    rank: int,
    device_count: int,
    device_id: Any,
) -> Tuple["Model", Set[str]]:
    
    sharding_strategy = ParameterShardingStrategy(device_count, rank)
    sharded_model, modified_parameters = sharding_strategy.shard_model_parameters(
        replica_model, config, device_id
    )
    return sharded_model, modified_parameters