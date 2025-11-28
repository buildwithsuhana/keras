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
    """
    Parameter-level sharding strategy that works with any Keras model.
    """

    def __init__(self, device_count: int, rank: int):
        self.device_count = device_count
        self.rank = rank
        self.sharded_weights_by_id = {}

    def shard_model_parameters(
        self,
        replica_model: "Model",
        original_model: "Model",
        config: LayoutMap,
        device_id: Any,
    ) -> Tuple["Model", Set[str]]:
        """
        Shard model parameters by modifying the replica_model in-place.
        """
        # We delay the import to avoid circular dependency
        ParameterShardedModel = _define_parameter_sharded_model()

        print(f"🔧 Applying parameter-level sharding to replica for Rank {self.rank}")
        
        # 1. Map Original Weights for easy lookup
        # We need to map {param_name: original_weight_value}
        original_weights_map = {}
        for weight in original_model.weights:
            # We use the weight name to match. clone_model preserves names.
            # Clean name removes the backend specific ":0" suffix if present
            clean_name = weight.name.split(":")[0]
            if hasattr(weight, 'value'):
                original_weights_map[clean_name] = weight.value
            else:
                original_weights_map[clean_name] = weight
        
        modified_parameters = set()

        # 2. Iterate through specific layers in the config and apply sharding
        for pattern, action in config.state_rules.items():
            if callable(action):
                # Find matching parameters in the REPLICA model
                matching_params = self._find_matching_layers_and_attrs(replica_model, pattern)

                for layer, attr_name, param in matching_params:
                    param_name = param.name.split(":")[0]
                    
                    # Get the original value to slice
                    # If the replica name matches the original, we can look it up
                    if param_name in original_weights_map:
                        original_value = original_weights_map[param_name]
                    else:
                        # Fallback: try to find by fuzzy match or assumption
                        # This happens if clone_model changed names slightly (e.g. _1 suffix)
                        # For now, we assume strict name matching from clone_model
                        logger.warning(f"Could not find original weight for {param_name}, skipping sharding.")
                        continue

                    # Execute the lambda to get the shard (slice)
                    # This happens on CPU (assuming original_value is on CPU)
                    try:
                        sharded_value = action(original_value, self.rank)
                    except Exception as e:
                        logger.error(f"Error slicing {param_name}: {e}")
                        continue

                    # Create the new Variable directly on the target device
                    # This ensures the full tensor is NEVER materialized on the Accelerator
                    with device(device_id):
                        new_sharded_var = Variable(
                            initializer=sharded_value,
                            trainable=param.trainable,
                            name=param.name,
                            dtype=param.dtype
                        )
                    
                    # 3. CRITICAL: Replace the variable in the replica layer
                    # We overwrite the attribute (e.g., layer.kernel = new_var)
                    setattr(layer, attr_name, new_sharded_var)
                    
                    modified_parameters.add(param_name)
                    print(
                        f"   ✅ Sharded {param_name}: {param.shape} -> {new_sharded_var.shape} on {device_id}"
                    )
                    
                    # Clean up the CPU slice immediately
                    del sharded_value

        # Wrap the modified replica
        sharded_model_wrapper = ParameterShardedModel(
            model_shard=replica_model,
            original_model_ref=original_model, # Only for config/metadata
            sharding_strategy=self,
            config=config,
            device_id=device_id,
        )

        print(
            f"🎯 Parameter sharding completed: {len(modified_parameters)} parameters sharded"
        )
        return sharded_model_wrapper, modified_parameters

    def _find_matching_layers_and_attrs(self, model, pattern: str) -> List[Tuple[layers.Layer, str, Any]]:
        """
        Find (layer, attribute_name, weight_variable) matching the pattern.
        """
        matches = []
        processed_objs = set()

        def search_recursive(obj, prefix=""):
            if id(obj) in processed_objs:
                return
            processed_objs.add(id(obj))

            # If it's a layer/model, check its weights
            if isinstance(obj, (layers.Layer, keras.Model)):
                name = obj.name
                full_name = f"{prefix}.{name}" if prefix else name
                
                # Check direct attributes that are weights
                # We iterate dir() to find which attribute holds the weight
                # This is safer than obj.weights which loses the attribute name context
                for attr_name in dir(obj):
                    if attr_name.startswith("_"): continue
                    
                    try:
                        val = getattr(obj, attr_name)
                    except Exception:
                        continue
                        
                    if isinstance(val, Variable):
                        # Construct the full parameter name to match regex
                        # Typically: model.layer.kernel
                        cleaned_weight_name = val.name.split("/")[-1].split(":")[0]
                        
                        # Match against the config pattern
                        # Pattern usually matches "layer_name.kernel"
                        # We construct "layer_full_path.attr_name" or just check existing pattern logic
                        # The config patterns from autoconfig are usually "full_layer_path.kernel"
                        
                        # We try to reconstruct the name expected by the pattern
                        candidate_name = f"{full_name}.{cleaned_weight_name}"
                        
                        if re.fullmatch(pattern, candidate_name):
                             matches.append((obj, attr_name, val))
                    
                    # Recurse into sub-layers
                    if isinstance(val, (layers.Layer, keras.Model)):
                        search_recursive(val, full_name)
                    elif isinstance(val, (list, tuple)):
                        for item in val:
                            if isinstance(item, (layers.Layer, keras.Model)):
                                search_recursive(item, full_name)

        search_recursive(model, prefix="")
        return matches


def _define_parameter_sharded_model():
    """
    Factory function to define and return the ParameterShardedModel class.
    """
    from keras.src.models import Model
    from keras.src.models import Functional

    class ParameterShardedModel(Model):
        """
        Wrapper that drives a specific Model Shard (Replica).
        """

        def __init__(
            self,
            model_shard: Model,
            original_model_ref: Model,
            sharding_strategy: ParameterShardingStrategy,
            config: LayoutMap,
            device_id: Any,
        ):
            # We assume the name is same
            super().__init__(name=model_shard.name)

            self.model_shard = model_shard
            self.original_model_ref = original_model_ref # Keep ref just for get_config if needed
            self.sharding_strategy = sharding_strategy
            self.config = config
            self._device = device_id

            # Ensure the shard is built
            if not self.model_shard.built and self.original_model_ref.inputs:
                try:
                    self.model_shard.build(self.original_model_ref.inputs[0].shape)
                except:
                    pass

            print("🚀 ParameterShardedModel created wrapping a unique replica")

        @property
        def device(self):
            return self._device

        @property
        def weights(self):
            return self.model_shard.weights
        
        @property
        def trainable_weights(self):
            return self.model_shard.trainable_weights

        @property
        def non_trainable_weights(self):
            return self.model_shard.non_trainable_weights

        def compute_output_shape(self, input_shape):
            return self.model_shard.compute_output_shape(input_shape)

        def compute_output_spec(self, *args, **kwargs):
            try:
                return self.model_shard.compute_output_spec(*args, **kwargs)
            except:
                 return self.original_model_ref.compute_output_spec(*args, **kwargs)

        def call(self, inputs, training=None, mask=None):
            # We are running on a specific device (managed by TensorParallelKeras context)
            # We act as a coordinator for this shard.

            # We need to map inputs to what the replica expects.
            # Since replica is a clone, it expects same input structure.
            
            # 1. Cache Inputs for intermediate layer lookup
            tensor_cache = {}
            flat_inputs = ops.convert_to_tensor(inputs) if not isinstance(inputs, dict) else inputs
            
            # Simple caching strategy
            if isinstance(inputs, dict):
                for k, v in inputs.items():
                    tensor_cache[k] = v
            
            # 2. Iterate Layers of the REPLICA
            # We must execute the layers of the `model_shard` because they hold the SHARDED weights.
            for layer in self.model_shard.layers:
                if isinstance(layer, layers.InputLayer):
                    continue

                # Reconstruct inputs
                layer_inputs = []
                # Inbound nodes of the replica
                for node in layer._inbound_nodes:
                    for input_tensor in node.input_tensors:
                        # We need to find the value for this tensor.
                        # Since we are executing the replica, we track the replica's tensors.
                        
                        # Lookup by ID
                        if id(input_tensor) in tensor_cache:
                            layer_inputs.append(tensor_cache[id(input_tensor)])
                            continue
                        
                        # Lookup by name (fallback)
                        clean_name = input_tensor.name.split(":")[0]
                        if clean_name in tensor_cache:
                            layer_inputs.append(tensor_cache[clean_name])
                            continue
                        
                        # If not found, it might be a model input
                        # Try matching with model_shard.inputs
                        found = False
                        for idx, model_in in enumerate(self.model_shard.inputs):
                            if model_in.name == input_tensor.name:
                                # We assume 'inputs' arg matches model inputs order/structure
                                if isinstance(inputs, (list, tuple)):
                                    layer_inputs.append(inputs[idx])
                                elif isinstance(inputs, dict):
                                    layer_inputs.append(inputs[model_in.name.split(":")[0]])
                                else:
                                    layer_inputs.append(inputs)
                                found = True
                                break
                        if not found:
                             # Fallback: if we simply can't find it, proceed (might crash or be handled below)
                             pass

                # Handle Dict inputs/Single inputs
                if len(layer_inputs) == 1:
                    layer_inputs = layer_inputs[0]
                elif hasattr(layer, "input_names") and len(layer_inputs) == len(layer.input_names):
                    layer_inputs = dict(zip(layer.input_names, layer_inputs))

                if not layer_inputs:
                     # Edge case: Layer connected directly to inputs but logic failed
                     layer_inputs = inputs

                # CALL THE LAYER (This uses the SHARDED weights on GPU)
                # Pass training flag
                call_kwargs = {}
                if training is not None:
                    call_kwargs['training'] = training
                
                # Check for mask support
                if layer.supports_masking and mask is not None:
                     call_kwargs['mask'] = mask

                current_tensor = layer(layer_inputs, **call_kwargs)

                # 3. Apply Communication (AllReduce/AllGather)
                # We check the config based on the Original Model's structure/names
                # Fortunately, clone_model preserves names.
                layer_path = layer.path if hasattr(layer, 'path') else layer.name
                
                output_rule = None
                for pattern, rule in self.config.output_rules.items():
                    if re.search(pattern, layer_path):
                        output_rule = rule.get(0) # Rule for rank 0 (applied symmetrically usually)
                        break

                if output_rule:
                    current_tensor = self._apply_communication(
                        current_tensor, layer.name, output_rule
                    )

                # 4. Update Cache
                # We map the OUTPUTS of this layer node to the result
                for node in layer._inbound_nodes:
                     outputs = getattr(node, "output_tensors", [])
                     if not isinstance(outputs, (list, tuple)):
                         outputs = [outputs]
                     
                     if isinstance(current_tensor, (list, tuple)) and len(current_tensor) == len(outputs):
                         for out_t, res in zip(outputs, current_tensor):
                             tensor_cache[id(out_t)] = res
                             tensor_cache[out_t.name.split(":")[0]] = res
                     else:
                         for out_t in outputs:
                             tensor_cache[id(out_t)] = current_tensor
                             tensor_cache[out_t.name.split(":")[0]] = current_tensor

            # Return final outputs matching model_shard.outputs
            final_outputs = []
            for out_t in self.model_shard.outputs:
                if id(out_t) in tensor_cache:
                    final_outputs.append(tensor_cache[id(out_t)])
                elif out_t.name.split(":")[0] in tensor_cache:
                    final_outputs.append(tensor_cache[out_t.name.split(":")[0]])
                else:
                    raise ValueError(f"Could not find output {out_t.name} in tensor cache")
            
            if len(final_outputs) == 1:
                return final_outputs[0]
            return final_outputs

        def _apply_communication(self, sharded_output, layer_name, rule_str: str):
            """Applies communication directly using the distributed backend."""
            if "sum" in rule_str or "allreduce" in rule_str:
                return distribution_lib.all_reduce(
                    sharded_output, op="sum", axis_name="model"
                )
            elif "gather" in rule_str:
                try:
                    parts = rule_str.split(" ")
                    dim = int(parts[-1]) if len(parts) > 1 else -1
                except:
                    dim = -1
                return distribution_lib.all_gather(
                    sharded_output, axis=dim, axis_name="model"
                )
            return sharded_output

        def get_config(self):
            return self.original_model_ref.get_config()

        @classmethod
        def from_config(cls, config, custom_objects=None):
            return cls(**config)

    return ParameterShardedModel


def make_parameter_sharded_model(
    model: "Model",
    config: LayoutMap,
    rank: int,
    device_count: int,
    device_id: Any,
) -> Tuple["Model", Set[str]]:
    """
    Create a parameter-sharded version of a Keras model.
    """
    # 1. Clone the model on CPU first to avoid OOM during initialization
    # We want the 'skeleton' to be on CPU.
    with device("cpu"):
        # clone_model creates new variables. We will overwrite them shortly.
        try:
            replica_model = keras.models.clone_model(model)
        except Exception as e:
            logger.warning(f"clone_model failed ({e}), attempting to use original model structure (Not Recommended for TP)")
            replica_model = model

    sharding_strategy = ParameterShardingStrategy(device_count, rank)

    # 2. Shard the replica by pulling from 'model' (CPU) and pushing to 'device_id' (GPU)
    sharded_model, modified_parameters = sharding_strategy.shard_model_parameters(
        replica_model, model, config, device_id
    )

    return sharded_model, modified_parameters