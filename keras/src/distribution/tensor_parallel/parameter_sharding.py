import logging
import inspect
import re
import functools
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

import numpy as np

from keras.src.backend import distribution_lib
from keras.src import ops
from keras.src import layers
from keras import Variable, device 

from keras.src.distribution.tensor_parallel.tensor_layout import LayoutMap, split_tensor_for_parallelism

if TYPE_CHECKING:
    from keras.src.models import Model

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

        safe_name = name.replace("/", "_").replace(":", "_")

        with device(current_dev_name):
            self._variable = Variable(
                initializer=tensor_shard, trainable=trainable, name=safe_name
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
        self.param_path_map = {}
        self._id_to_param_map = {}

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

        self.param_path_map = {w.path: w for w in model.weights}
        
        for w in model.weights:
            self._id_to_param_map[id(w)] = (w.path, w)
            exp_ref_fn = getattr(w, "experimental_ref", None)
            if exp_ref_fn:
                self._id_to_param_map[id(exp_ref_fn())] = (w.path, w)
        
        # --- FIX: Discover missing Embeddings (e.g. PositionEmbedding) to reach 122 ---
        for layer in model._flatten_layers(recursive=True, include_self=True):
            if "Embedding" in layer.__class__.__name__:
                for attr in ["embeddings", "position_embeddings", "_embeddings"]:
                    var = getattr(layer, attr, None)
                    if var is not None:
                        var_id = id(getattr(var, "experimental_ref", lambda: var)())
                        if var_id not in config.state_rules:
                            # Shard along embedding dim (1) using same order as autoconfig
                            config.state_rules[var_id] = functools.partial(
                                split_tensor_for_parallelism, 
                                device_count=self.device_count, 
                                dim=1
                            )

        # --- FIX: Normalize config keys (ID -> Path) to prevent library crashes ---
        normalized_updates = {}
        for pattern, action in list(config.state_rules.items()):
            if isinstance(pattern, int) and pattern in self._id_to_param_map:
                path, _ = self._id_to_param_map[pattern]
                normalized_updates[path] = action
                del config.state_rules[pattern]
        config.state_rules.update(normalized_updates)

        self._store_original_weights(model)
        modified_parameters = set()

        for pattern, action in config.state_rules.items():
            if callable(action):
                matching_params = self._find_matching_parameters(model, pattern)

                for param_name, param in matching_params:
                    exp_ref_fn = getattr(param, "experimental_ref", None)
                    param_id = id(exp_ref_fn()) if exp_ref_fn else id(param)

                    if param_id in self.sharded_weights_by_id:
                        self.sharded_weights[param_name] = self.sharded_weights_by_id[param_id]
                        modified_parameters.add(param_name)
                        continue

                    sharded_tensor = action(param, self.rank)
                    self.sharded_weights[param_name] = sharded_tensor
                    self.sharded_weights_by_id[param_id] = sharded_tensor
                    self.weight_mapping[param_name] = {
                        "original_shape": param.shape,
                        "sharded_shape": sharded_tensor.shape,
                        "action": action,
                    }
                    modified_parameters.add(param_name)
                    print(f"   ✅ Sharded {param_name}: {param.shape} -> {sharded_tensor.shape}")

        sharded_model = ParameterShardedModel(
            original_model=model,
            sharding_strategy=self,
            config=config,
            device_id=device_id,
        )

        print(f"🎯 Parameter sharding completed: {len(modified_parameters)} parameters sharded")
        return sharded_model, modified_parameters

    def _store_original_weights(self, model):
        for weight in model.weights:
            if hasattr(weight, 'numpy'):
                self.original_weights[weight.path] = weight.numpy()

    def _find_matching_parameters(self, model, pattern: Any) -> List[Tuple[str, Any]]:
        if isinstance(pattern, int):
            if pattern in self._id_to_param_map:
                return [self._id_to_param_map[pattern]]
            return []
        if not isinstance(pattern, str):
            return []
        if pattern in self.param_path_map:
            return [(pattern, self.param_path_map[pattern])]
        matches = []
        suffix = "/" + pattern
        for path, weight in self.param_path_map.items():
            if path.endswith(suffix):
                matches.append((path, weight))
        return matches


def _define_parameter_sharded_model():
    from keras.src.models import Model, Functional

    class ParameterShardedModel(Model):
        def __init__(self, original_model: Model, sharding_strategy: ParameterShardingStrategy, config: LayoutMap, device_id: Any):
            super().__init__(name=original_model.name)
            self.original_model = original_model
            self.sharding_strategy = sharding_strategy
            self.config = config
            self._device = device_id
            if not self.original_model.built and self.original_model.inputs:
                self.original_model.build(self.original_model.inputs[0].shape)
            self._build_and_cache_weights()
            print("🚀 ParameterShardedModel created successfully")

        @property
        def device(self):
            return self._device

        def _build_and_cache_weights(self):
            weights_list = []
            sharded_weight_ids = set(self.sharding_strategy.sharded_weights_by_id.keys())
            for param_name, sharded_tensor in self.sharding_strategy.sharded_weights.items():
                weights_list.append(ShardedWeight(sharded_tensor, param_name, device_id=self._device))
            for weight in self.original_model.weights:
                exp_ref_fn = getattr(weight, "experimental_ref", None)
                weight_id = id(exp_ref_fn()) if exp_ref_fn else id(weight)
                if weight_id not in sharded_weight_ids:
                    weights_list.append(weight)
            self._weights_list = weights_list

        @property
        def weights(self):
            return self._weights_list
        
        def compute_output_shape(self, input_shape):
            return self.original_model.compute_output_shape(input_shape)

        def compute_output_spec(self, *args, **kwargs):
            if args:
                return self.original_model.compute_output_spec(args[0])
            return self.original_model.compute_output_spec(**kwargs)

        def call(self, inputs, training=None, mask=None):
            tensor_cache = {}
            # 1. Cache Inputs
            if isinstance(inputs, dict):
                for inp_tensor in self.original_model.inputs:
                    name = getattr(inp_tensor, "name", None)
                    if name:
                        clean_name = name.split(":")[0]
                        val = inputs.get(name, inputs.get(clean_name))
                        if val is not None:
                            tensor_cache[id(inp_tensor)] = val
                            tensor_cache[clean_name] = val
                            tensor_cache[name] = val
            else:
                input_list = inputs if isinstance(inputs, (list, tuple)) else [inputs]
                for i, inp_tensor in enumerate(self.original_model.inputs):
                    if i < len(input_list):
                        val = input_list[i]
                        tensor_cache[id(inp_tensor)] = val
                        name = getattr(inp_tensor, "name", None)
                        if name:
                            tensor_cache[name.split(":")[0]] = val
                            tensor_cache[name] = val

            # 2. Iterate Layers
            for layer in self.original_model.layers:
                if isinstance(layer, layers.InputLayer):
                    continue
                
                layer_inputs = []
                for node in layer._inbound_nodes:
                    for symbolic_input_tensor in node.input_tensors:
                        val = tensor_cache.get(id(symbolic_input_tensor))
                        if val is None:
                            name = getattr(symbolic_input_tensor, "name", None)
                            if name:
                                val = tensor_cache.get(name.split(":")[0], tensor_cache.get(name))
                        if val is not None:
                            layer_inputs.append(val)
                
                # --- FIX: Reconstruct Dict for Backbones to resolve structural mismatch ---
                if isinstance(layer, (Functional, Model)):
                    layer_input_names = getattr(layer, "input_names", [])
                    if not layer_input_names and hasattr(layer, "inputs") and layer.inputs:
                        layer_input_names = [getattr(x, "name", "").split(":")[0] for x in layer.inputs]
                    
                    if layer_input_names and len(layer_inputs) == len(layer_input_names):
                        layer_inputs = dict(zip(layer_input_names, layer_inputs))
                elif len(layer_inputs) == 1:
                    layer_inputs = layer_inputs[0]

                if not layer_inputs and (isinstance(layer_inputs, (list, tuple, dict)) or layer_inputs is None):
                    layer_inputs = inputs

                call_kwargs = {"training": training} if training is not None else {}
                node_args = getattr(node, "arguments", None)
                if node_args:
                    extra_kwargs = getattr(node_args, "kwargs", {})
                    if extra_kwargs:
                        for k, v in extra_kwargs.items():
                            if k != "training":
                                call_kwargs[k] = v

                current_tensor = layer(layer_inputs, **call_kwargs)

                # 4. Apply Communication Rules
                layer_path = getattr(layer, "path", layer.name)
                output_rule = None
                for pattern, rule in self.config.output_rules.items():
                    if pattern == layer_path or layer_path.endswith("/" + pattern) or re.search(str(pattern), layer_path):
                        output_rule = rule.get(0) if isinstance(rule, dict) else rule
                        break

                if output_rule:
                    current_tensor = output_rule(current_tensor) if callable(output_rule) else self._apply_communication(current_tensor, layer.name, output_rule)

                # 5. Map Outputs
                for node_idx, node in enumerate(layer._inbound_nodes):
                    outputs = getattr(node, "output_tensors", None)
                    if not outputs:
                        continue
                    vals = current_tensor if isinstance(current_tensor, (list, tuple)) else [current_tensor]
                    for out_idx, (sym, val) in enumerate(zip(outputs, vals)):
                        tensor_cache[id(sym)] = val
                        tensor_cache[("layer_node_output", layer.name, node_idx, out_idx)] = val
                        name = getattr(sym, "name", None)
                        if name:
                            tensor_cache[name.split(":")[0]] = val
                            tensor_cache[name] = val

            final_outputs = []
            for symbolic_output in self.original_model.outputs:
                val = tensor_cache.get(id(symbolic_output))
                if val is None:
                    name = getattr(symbolic_output, "name", None)
                    if name:
                        val = tensor_cache.get(name.split(":")[0], tensor_cache.get(name))
                if val is None:
                    hist = getattr(symbolic_output, "_keras_history", None)
                    if hist and len(hist) >= 3:
                        val = tensor_cache.get(("layer_node_output", getattr(hist[0], "name", None), hist[1], hist[2]))
                if val is None:
                    raise RuntimeError(f"Missing runtime value for output: {symbolic_output}.")
                final_outputs.append(val)

            return final_outputs[0] if len(final_outputs) == 1 else final_outputs

        def _apply_communication(self, sharded_output, layer_name, rule_str: str):
            if "sum" in rule_str or "allreduce" in rule_str:
                return distribution_lib.all_reduce(sharded_output, op="sum", axis_name="model")
            if "gather" in rule_str:
                parts = rule_str.split(" ")
                dim = int(parts[-1]) if len(parts) > 1 and parts[-1].lstrip('-').isdigit() else -1
                return distribution_lib.all_gather(sharded_output, axis=dim, axis_name="model")
            return sharded_output

        def get_config(self):
            return self.original_model.get_config()

        @classmethod
        def from_config(cls, config, custom_objects=None):
            return cls(**config)

    return ParameterShardedModel


def make_parameter_sharded_model(module: "Model", config: LayoutMap, rank: int, device_count: int, device_id: Any) -> Tuple["Model", Set[str]]:
    sharding_strategy = ParameterShardingStrategy(device_count, rank)
    return sharding_strategy.shard_model_parameters(module, config, device_id)