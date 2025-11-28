import logging
import re
import gc
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

import numpy as np
import keras
from keras import Variable, device 
from keras.src.backend import distribution_lib
from keras.src import ops
from keras.src import layers

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
        print(f"   🔧 Sharding parameters for Rank {self.rank} on {device_id}...")
        
        modified_parameters = set()
        
        # Flatten the loop to allow easier GC
        all_matches = []
        for pattern, action in config.state_rules.items():
            if callable(action):
                matches = self._find_matching_layers_and_attrs(replica_model, pattern)
                for m in matches:
                    all_matches.append((m, action))

        total_layers = len(all_matches)
        
        for idx, ((layer, attr_name, param), action) in enumerate(all_matches):
            param_name = param.name
            
            # 1. Extract CPU Value (Keep as numpy to avoid backend memory holding)
            # Use .numpy() directly to detach from graph immediately
            try:
                original_value = param.value.numpy()
            except AttributeError:
                original_value = param.value
            
            # 2. Slice (Perform on CPU)
            try:
                # Ensure we don't accidentally promote bfloat16 to float32 if not needed
                sharded_value = action(original_value, self.rank)
            except Exception as e:
                logger.error(f"Error slicing {param_name}: {e}")
                del original_value
                continue

            # 3. Create GPU Variable
            # We create it directly on the target device
            with device(device_id):
                new_sharded_var = Variable(
                    initializer=sharded_value,
                    trainable=param.trainable,
                    name=param_name,
                    dtype=param.dtype 
                )
            
            # 4. Replace in Layer (Swap CPU weight for GPU weight)
            setattr(layer, attr_name, new_sharded_var)
            modified_parameters.add(param_name)
            
            # 5. AGGRESSIVE CLEANUP
            # Delete the heavy numpy arrays immediately
            del original_value
            del sharded_value
            
            # Manually trigger GC every 20 layers to prevent RAM spikes
            if idx % 20 == 0:
                gc.collect()

        # Clear any internal Keras caches that might hold references to old weights
        if hasattr(replica_model, '_trainable_variables'):
            del replica_model._trainable_variables
        if hasattr(replica_model, '_non_trainable_variables'):
            del replica_model._non_trainable_variables

        # Final massive GC before returning
        gc.collect()

        sharded_model_wrapper = ParameterShardedModel(
            model_shard=replica_model,
            sharding_strategy=self,
            config=config,
            device_id=device_id,
        )
        return sharded_model_wrapper, modified_parameters

    def _find_matching_layers_and_attrs(self, model, pattern: str):
        """Find (layer, attribute_name, weight_variable) matching the pattern."""
        matches = []
        processed_objs = set()

        def search_recursive(obj, prefix=""):
            if id(obj) in processed_objs: return
            processed_objs.add(id(obj))

            if isinstance(obj, (layers.Layer, keras.Model)):
                name = obj.name
                full_name = f"{prefix}.{name}" if prefix else name
                
                # Iterate all attributes to find weights
                for attr_name in dir(obj):
                    if attr_name.startswith("_"): continue
                    try:
                        val = getattr(obj, attr_name)
                    except: continue
                        
                    if isinstance(val, Variable):
                        cleaned = val.name.split("/")[-1].split(":")[0]
                        candidate = f"{full_name}.{cleaned}"
                        # Match regex
                        if re.fullmatch(pattern, candidate):
                             matches.append((obj, attr_name, val))
                    
                    # Recurse
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
        def __init__(self, model_shard, sharding_strategy, config, device_id):
            super().__init__(name=model_shard.name)
            self.model_shard = model_shard
            self.sharding_strategy = sharding_strategy
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
            # 1. Forward pass on the shard
            # The shard already contains the GPU weights, so this executes on GPU
            output = self.model_shard(inputs, training=training, mask=mask)
            
            # 2. Apply Output Communication Rules (AllReduce/AllGather)
            # We check the rule for the *last layer* of the shard to see if we need to sync
            # Note: For deep models, internal layers might have already applied rules 
            # if they were wrapped or if we inserted communication ops. 
            # For this implementation, we assume the config handles internal splits via 
            # the sharded layers themselves (if implemented) or we rely on the final aggregation.
            
            # Check for output rules for the last layer
            try:
                last_layer_name = self.model_shard.layers[-1].name
                rule = self.config.output_rules.get(last_layer_name)
                
                # If exact name match failed, try regex
                if not rule:
                    for pattern, r in self.config.output_rules.items():
                        if re.search(pattern, last_layer_name):
                            rule = r
                            break
                            
                if rule:
                    rule_str = rule.get(0) # Rule for rank 0 applies to all usually
                    if rule_str:
                        if "sum" in rule_str or "allreduce" in rule_str:
                            output = distribution_lib.all_reduce(output, op="sum", axis_name="model")
                        elif "gather" in rule_str:
                            # Parse axis if present (e.g. "gather -1")
                            axis = -1
                            parts = rule_str.split()
                            if len(parts) > 1:
                                try: axis = int(parts[1])
                                except: pass
                            output = distribution_lib.all_gather(output, axis=axis, axis_name="model")
            except Exception as e:
                # Fallback or ignore if structure is complex
                pass

            return output

    return ParameterShardedModel

def make_parameter_sharded_model(replica_model, config, rank, device_count, device_id):
    strategy = ParameterShardingStrategy(device_count, rank)
    return strategy.shard_model_parameters(replica_model, config, device_id)