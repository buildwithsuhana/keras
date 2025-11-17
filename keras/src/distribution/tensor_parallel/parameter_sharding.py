import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from keras.src.backend import distribution_lib
from keras.src import ops
from keras.src import layers
from keras import Variable, device 

from keras.src.distribution.tensor_parallel.tensor_layout import LayoutMap

logger = logging.getLogger(__name__)


class ShardedWeight:
    """
    A wrapper for a sharded Keras Variable, providing a consistent interface.
    """

    def __init__(self, tensor_shard, name, trainable=True, device_id=None):
        current_dev_name = device_id if device_id else "UNKNOWN_DEVICE"
        print(
            f"   [DEV: {current_dev_name}] 🧬 Creating Sharded Variable "
            f"'{name}' with shape {tensor_shard.shape}"
        )

        with device(current_dev_name):
            self._variable = Variable(
                initializer=tensor_shard, trainable=trainable, name=name
            )
        self.regularizer = None

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

    def num_elements(self):
        return ops.size(self._variable)

    def __repr__(self):
        return (
            f"<ShardedWeight name='{self.name}' "
            f"shape={self.shape} trainable={self.trainable}>"
        )


class ParameterShardingStrategy:
    """
    Parameter-level sharding strategy that works with any Keras model.
    """

    def __init__(self, device_count: int, rank: int):
        self.device_count = device_count
        self.rank = rank
        self.sharded_weights = {}
        self.original_weights = {}
        self.weight_mapping = {}
        self.sharded_weights_by_id = {}

    def shard_model_parameters(
        self,
        model,
        config: LayoutMap,
        device_id: Any,
    ) -> Tuple["Model", Set[str]]:
        """
        Shard model parameters without rebuilding the model structure.
        """
        ParameterShardedModel = _define_parameter_sharded_model()

        print(f"🔧 Applying parameter-level sharding to {model.name}")

        self._store_original_weights(model)
        modified_parameters = set()

        for pattern, action in config.state_rules.items():
            # FIX: Check if action is callable (lambda) instead of isinstance(Split)
            if callable(action):
                matching_params = self._find_matching_parameters(model, pattern)

                for param_name, param in matching_params:
                    try:
                        param_id = id(param.experimental_ref())
                    except AttributeError:
                        param_id = id(param)

                    # Weight tying logic
                    if param_id in self.sharded_weights_by_id:
                        self.sharded_weights[param_name] = self.sharded_weights_by_id[
                            param_id
                        ]
                        
                        # Find existing name for logging/mapping
                        existing_param_name = "unknown"
                        for name, shard in self.sharded_weights.items():
                            if shard is self.sharded_weights_by_id[param_id]:
                                existing_param_name = name
                                break

                        self.weight_mapping[param_name] = self.weight_mapping.get(
                            existing_param_name, {}
                        )
                        modified_parameters.add(param_name)
                        print(
                            f"   🔗 Tied {param_name} to existing shard from {existing_param_name}"
                        )
                        continue

                    # Execute the lambda to get the shard (slice)
                    sharded_tensor = action(param, self.rank)

                    self.sharded_weights[param_name] = sharded_tensor
                    self.sharded_weights_by_id[param_id] = sharded_tensor

                    self.weight_mapping[param_name] = {
                        "original_shape": param.shape,
                        "sharded_shape": sharded_tensor.shape,
                        "action": action,
                    }

                    modified_parameters.add(param_name)
                    print(
                        f"   ✅ Sharded {param_name}: {param.shape} -> {sharded_tensor.shape}"
                    )

        sharded_model = ParameterShardedModel(
            original_model=model,
            sharding_strategy=self,
            config=config,
            device_id=device_id,
        )

        print(
            f"🎯 Parameter sharding completed: {len(modified_parameters)} parameters sharded"
        )
        return sharded_model, modified_parameters

    def _store_original_weights(self, model):
        """Store original weights for reference."""
        
        def find_weights_recursive(current_layer, prefix=""):
            name = current_layer.name
            full_name = f"{prefix}.{name}" if prefix else name

            if hasattr(current_layer, "weights") and current_layer.weights:
                for weight in current_layer.weights:
                    cleaned_name = weight.name.split("/")[-1].split(":")[0]
                    param_name = f"{full_name}.{cleaned_name}"
                    
                    # FIX: Check for numpy capability before access
                    if hasattr(weight, 'numpy'):
                        self.original_weights[param_name] = weight.numpy()

            if hasattr(current_layer, "layers") and current_layer.layers:
                for sub_layer in current_layer.layers:
                    find_weights_recursive(sub_layer, full_name)

            for attr_name in dir(current_layer):
                if attr_name.startswith("__") and attr_name.endswith("__"):
                    continue
                try:
                    attr = getattr(current_layer, attr_name)
                except Exception:
                    continue
                if isinstance(attr, layers.Layer) and attr is not current_layer:
                    find_weights_recursive(attr, full_name)
                elif isinstance(attr, (list, tuple)):
                    for item in attr:
                        if isinstance(item, layers.Layer):
                            find_weights_recursive(item, full_name)

        # Start recursion
        find_weights_recursive(model, prefix="")

    def _find_matching_parameters(self, model, pattern: str) -> List[Tuple[str, Any]]:
        """
        Find parameters that match the given pattern using smart recursion.
        """
        matching_params = []
        processed_layers = set()

        def search_layer_recursive(current_layer, prefix=""):
            if id(current_layer) in processed_layers:
                return
            processed_layers.add(id(current_layer))

            name = current_layer.name
            full_name = f"{prefix}.{name}" if prefix else name

            if hasattr(current_layer, "weights") and current_layer.weights:
                for weight in current_layer.weights:
                    cleaned_weight_name = weight.name.split("/")[-1].split(":")[0]
                    param_name = f"{full_name}.{cleaned_weight_name}"

                    # FIX: Use fullmatch for strict pattern adherence
                    if re.fullmatch(pattern, param_name):
                        matching_params.append((param_name, weight))

            if hasattr(current_layer, "layers") and current_layer.layers:
                for sub_layer in current_layer.layers:
                    search_layer_recursive(sub_layer, full_name)

            for attr_name in dir(current_layer):
                if attr_name.startswith("__") and attr_name.endswith("__"):
                    continue

                try:
                    attr = getattr(current_layer, attr_name)
                except Exception:
                    continue

                if isinstance(attr, layers.Layer) and attr is not current_layer:
                    search_layer_recursive(attr, full_name)

                elif isinstance(attr, (list, tuple)):
                    for item in attr:
                        if isinstance(item, layers.Layer):
                            search_layer_recursive(item, full_name)

        search_layer_recursive(model, prefix="")
        return matching_params


def _define_parameter_sharded_model():
    """
    Factory function to define and return the ParameterShardedModel class.
    This delays the import of keras.src.models.Model to break circular dependencies.
    """
    from keras.src.models import Model

    class ParameterShardedModel(Model):
        """
        Wrapper model that handles parameter sharding without rebuilding the structure.
        """

        def __init__(
            self,
            original_model: Model,
            sharding_strategy: ParameterShardingStrategy,
            config: LayoutMap,
            device_id: Any,
        ):
            # FIX: Initialize super with the original name
            super().__init__(name=original_model.name)

            self.original_model = original_model
            self.sharding_strategy = sharding_strategy
            self.config = config
            self._device = device_id

            # Build if not built (vital for inputs access)
            if not self.original_model.built and self.original_model.inputs:
                self.original_model.build(self.original_model.inputs[0].shape)

            self._build_and_cache_weights()

            if self.original_model.inputs:
                self.build(self.original_model.inputs[0].shape)

            print("🚀 ParameterShardedModel created successfully")

        @property
        def device(self):
            return self._device

        def _build_and_cache_weights(self):
            """
            Builds the list of trainable/non-trainable weights ONCE and caches it.
            """
            print("   - Building and caching the definitive weights list...")
            weights_list = []

            sharded_weight_ids = set(
                self.sharding_strategy.sharded_weights_by_id.keys()
            )

            for (
                param_name,
                sharded_tensor,
            ) in self.sharding_strategy.sharded_weights.items():
                weights_list.append(
                    ShardedWeight(
                        sharded_tensor, 
                        param_name, 
                        device_id=self._device  # FIX: Pass device ID correctly
                    )
                )

            unsharded_count = 0
            for weight in self.original_model.weights:
                try:
                    weight_id = id(weight.experimental_ref())
                except AttributeError:
                    weight_id = id(weight)

                if weight_id not in sharded_weight_ids:
                    weights_list.append(weight)
                    unsharded_count += 1

            self._weights_list = weights_list

        @property
        def weights(self):
            return self._weights_list

        def call(self, inputs, training=None, mask=None):
            tensor_cache = {}

            if isinstance(inputs, dict):
                for inp_tensor in self.original_model.inputs:
                    tensor_cache[id(inp_tensor)] = inputs[inp_tensor.name]
            else:
                tensor_cache[id(self.original_model.inputs[0])] = inputs

            for layer in self.original_model.layers:
                if isinstance(layer, layers.InputLayer):
                    continue
                
                layer_inputs = []
                for node in layer._inbound_nodes:
                    for symbolic_input_tensor in node.input_tensors:
                        layer_inputs.append(tensor_cache[id(symbolic_input_tensor)])
                
                if len(layer_inputs) == 1:
                    layer_inputs = layer_inputs[0]

                current_tensor = layer(layer_inputs, training=training)

                # Cache pre-communication tensor
                tensor_cache[id(layer.output)] = current_tensor
                
                # Apply output rules (Communication)
                layer_path = layer.path
                output_rule = None
                for pattern, rule in self.config.output_rules.items():
                    if re.search(pattern, layer_path):
                        output_rule = rule.get(0)
                        break
                
                if output_rule:
                    current_tensor = self._apply_communication(
                        current_tensor, layer.name, output_rule
                    )
                    tensor_cache[id(layer.output)] = current_tensor

            # Return final outputs
            final_outputs = []
            for symbolic_output in self.original_model.outputs:
                final_outputs.append(tensor_cache[id(symbolic_output)])

            if len(final_outputs) == 1:
                return final_outputs[0]
            return final_outputs

        def _apply_communication(self, sharded_output, layer_name, rule_str: str):
            """Applies communication directly using the distributed backend."""

            if "sum" in rule_str or "allreduce" in rule_str:
                logger.debug(f"Applying AllReduce (sum) to {layer_name}")
                return distribution_lib.all_reduce(
                    sharded_output, op="sum", axis_name="model"
                )

            elif "gather" in rule_str:
                # Robust parsing for "gather -1" or "gather dim -1"
                try:
                    parts = rule_str.split(" ")
                    if len(parts) > 1:
                        dim = int(parts[-1])
                    else:
                        dim = -1
                except (ValueError, IndexError):
                    dim = -1
                
                logger.debug(f"Applying AllGather (dim={dim}) to {layer_name}")
                return distribution_lib.all_gather(
                    sharded_output, axis=dim, axis_name="model"
                )

            else:
                return sharded_output

        def get_config(self):
            return self.original_model.get_config()

        @classmethod
        def from_config(cls, config, custom_objects=None):
            return cls(**config)

    return ParameterShardedModel


def make_parameter_sharded_model(
    module: "Model",
    config: LayoutMap,
    rank: int,
    device_count: int,
    device_id: Any,
) -> Tuple["Model", Set[str]]:
    """
    Create a parameter-sharded version of a Keras model.
    """
    sharding_strategy = ParameterShardingStrategy(device_count, rank)

    sharded_model, modified_parameters = sharding_strategy.shard_model_parameters(
        module, config, device_id
    )

    return sharded_model, modified_parameters