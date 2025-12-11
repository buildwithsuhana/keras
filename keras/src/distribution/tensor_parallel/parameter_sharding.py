import logging
import inspect
import re
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

import numpy as np

from keras.src.backend import distribution_lib
from keras.src import ops
from keras.src import layers
from keras import Variable, device 

from keras.src.distribution.tensor_parallel.tensor_layout import LayoutMap

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
        
        self._store_original_weights(model)
        modified_parameters = set()

        for pattern, action in config.state_rules.items():
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
        """Store original weights for reference using variable paths."""
        for weight in model.weights:
            if hasattr(weight, 'numpy'):
                self.original_weights[weight.path] = weight.numpy()

    def _find_matching_parameters(self, model, pattern: str) -> List[Tuple[str, Any]]:
        """
        Find parameters matching the pattern using the pre-computed path map.
        """
        # 1. Exact match (Ideal case: autoconfig path matches model path exactly)
        if pattern in self.param_path_map:
            return [(pattern, self.param_path_map[pattern])]
            
        # 2. Suffix match (Common case: autoconfig generated paths without top-level model prefix)
        matches = []
        suffix = "/" + pattern
        for path, weight in self.param_path_map.items():
            if path.endswith(suffix):
                matches.append((path, weight))
        
        return matches


def _define_parameter_sharded_model():
    """
    Factory function to define and return the ParameterShardedModel class.
    This delays the import of keras.src.models.Model to break circular dependencies.
    """
    from keras.src.models import Model
    from keras.src.models import Functional

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
            super().__init__(name=original_model.name)

            self.original_model = original_model
            self.sharding_strategy = sharding_strategy
            self.config = config
            self._device = device_id

            # Build if not built (vital for inputs access)
            if not self.original_model.built and self.original_model.inputs:
                self.original_model.build(self.original_model.inputs[0].shape)

            self._build_and_cache_weights()

            print("🚀 ParameterShardedModel created successfully")

        @property
        def device(self):
            return self._device

        def _build_and_cache_weights(self):
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
                        device_id=self._device
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
        
        # --- FIX: Robust Output Spec handling ---
        def compute_output_shape(self, input_shape):
            return self.original_model.compute_output_shape(input_shape)

        def compute_output_spec(self, *args, **kwargs):
            try:
                return self.original_model.compute_output_spec(args[0])
            except Exception:
                return self.original_model.compute_output_spec(*args, **kwargs)

        def call(self, inputs, training=None, mask=None):

            tensor_cache = {}

            # 1. Cache Inputs
            if isinstance(inputs, dict):
                for inp_tensor in self.original_model.inputs:
                    name = inp_tensor.name
                    clean_name = name.split(":")[0]
                    # Prefer exact name key, then cleaned name; store by id
                    if name in inputs:
                        val = inputs[name]
                    elif clean_name in inputs:
                        val = inputs[clean_name]
                    else:
                        val = None

                    if val is not None:
                        tensor_cache[id(inp_tensor)] = val
                        # also store by cleaned name so lookups can match
                        tensor_cache[clean_name] = val
                        tensor_cache[name] = val
            else:
                input_list = ops.convert_to_tensor(inputs)
                if not isinstance(input_list, (list, tuple)):
                    input_list = [input_list]
                
                for i, inp_tensor in enumerate(self.original_model.inputs):
                    if i < len(input_list):
                        val = input_list[i]
                        tensor_cache[id(inp_tensor)] = val
                        try:
                            name = inp_tensor.name
                            clean_name = name.split(":")[0]
                            tensor_cache[clean_name] = val
                            tensor_cache[name] = val
                        except Exception:
                            pass

            # 2. Iterate Layers
            for layer in self.original_model.layers:
                if isinstance(layer, layers.InputLayer):
                    continue
                
                # Reconstruct inputs for this layer from the cache
                layer_inputs = []
                for node in layer._inbound_nodes:
                    for symbolic_input_tensor in node.input_tensors:
                        key_id = id(symbolic_input_tensor)
                        if key_id in tensor_cache:
                            layer_inputs.append(tensor_cache[key_id])
                        else:
                            # Fallback: try matching by cleaned symbolic name
                            try:
                                name = symbolic_input_tensor.name
                                clean_name = name.split(":")[0]
                            except Exception:
                                name = None
                                clean_name = None

                            if clean_name and clean_name in tensor_cache:
                                layer_inputs.append(tensor_cache[clean_name])
                            elif name and name in tensor_cache:
                                layer_inputs.append(tensor_cache[name])
                
                # --- CRITICAL FIX: Handle Dictionary Inputs for Backbones ---
                if (
                    (isinstance(layer, Functional) or isinstance(layer, Model))
                    and hasattr(layer, "input_names")
                ):
                    if len(layer_inputs) == len(layer.input_names) and len(layer_inputs) > 0:
                        layer_inputs = dict(zip(layer.input_names, layer_inputs))
                
                elif len(layer_inputs) == 1:
                    layer_inputs = layer_inputs[0]

                try:
                    is_empty_container = (
                        layer_inputs is None
                        or (isinstance(layer_inputs, (list, tuple)) and len(layer_inputs) == 0)
                        or (isinstance(layer_inputs, dict) and len(layer_inputs) == 0)
                    )

                    if is_empty_container:
                        logger.debug(
                            "Layer '%s' received empty reconstructed inputs; forwarding top-level inputs",
                            layer.name,
                        )
                        layer_inputs = inputs

                    node_kwargs = {}
                    try:
                        if hasattr(node, "arguments") and getattr(node, "arguments") is not None:
                            node_kwargs = getattr(node.arguments, "kwargs", {}) or {}
                    except Exception:
                        node_kwargs = {}

                    call_kwargs = {"training": training} if training is not None else {}
                    for k, v in node_kwargs.items():
                        if k != "training":
                            call_kwargs[k] = v

                    current_tensor = layer(layer_inputs, **call_kwargs)
                except Exception:
                    tried_call = False

                    # (1) list/tuple -> dict by names
                    if isinstance(layer_inputs, (list, tuple)):
                        cleaned_names = None
                        try:
                            if hasattr(layer, "inputs") and layer.inputs:
                                cleaned_names = [n.name.split(":")[0] for n in layer.inputs]
                        except Exception:
                            cleaned_names = None

                        if not cleaned_names and hasattr(layer, "input_names"):
                            try:
                                cleaned_names = [n.split(":")[0] for n in layer.input_names]
                            except Exception:
                                cleaned_names = None

                        if cleaned_names and len(cleaned_names) == len(layer_inputs):
                            alt_inputs = dict(zip(cleaned_names, layer_inputs))
                            try:
                                logger.debug(
                                    "Retrying layer '%s' with dict inputs: %s",
                                    layer.name,
                                    list(alt_inputs.keys()),
                                )
                                current_tensor = layer(alt_inputs, training=training)
                                tried_call = True
                            except Exception:
                                tried_call = False

                    # (2) dict -> single tensor (first value)
                    if not tried_call and isinstance(layer_inputs, dict):
                        first_val = next(iter(layer_inputs.values()), None)
                        if first_val is not None:
                            try:
                                logger.debug(
                                    "Retrying layer '%s' with single tensor from dict inputs",
                                    layer.name,
                                )
                                current_tensor = layer(first_val, training=training)
                                tried_call = True
                            except Exception:
                                tried_call = False

                    # (3) list/tuple -> single positional tensor (first element)
                    if not tried_call and isinstance(layer_inputs, (list, tuple)) and len(layer_inputs) > 0:
                        try:
                            logger.debug(
                                "Retrying layer '%s' with first element of positional inputs",
                                layer.name,
                            )
                            current_tensor = layer(layer_inputs[0], training=training)
                            tried_call = True
                        except Exception:
                            tried_call = False

                    if not tried_call:
                        raise

                # 4/5. Apply Communication Rules (AllReduce/Gather) and Cache Output
                layer_path = getattr(layer, "path", layer.name)
                output_rule = None
                
                # Check for exact match or suffix match in output_rules
                for pattern, rule in self.config.output_rules.items():
                    if pattern == layer_path or layer_path.endswith("/" + pattern):
                        output_rule = rule.get(0)
                        break
                
                # Fallback: check regex if simple matching failed
                if not output_rule:
                    for pattern, rule in self.config.output_rules.items():
                        if re.search(pattern, layer_path):
                            output_rule = rule.get(0)
                            break

                if output_rule:
                    if callable(output_rule):
                        logger.debug(f"Applying callable Output Rule to {layer.name}")
                        current_tensor = output_rule(current_tensor)
                    else:
                        # Fallback for strings (legacy support)
                        current_tensor = self._apply_communication(
                            current_tensor, layer.name, output_rule
                        )

                # Map the runtime outputs back to the symbolic output tensors
                for node_idx, node in enumerate(layer._inbound_nodes):
                    try:
                        outputs = getattr(node, "output_tensors", None)
                    except Exception:
                        outputs = None

                    if not outputs:
                        continue

                    if isinstance(current_tensor, (list, tuple)):
                        for out_idx, (sym, val) in enumerate(zip(outputs, current_tensor)):
                            tensor_cache[id(sym)] = val
                            try:
                                layer_key = ("layer_node_output", layer.name, node_idx, out_idx)
                                tensor_cache[layer_key] = val
                            except Exception:
                                pass
                            try:
                                name = getattr(sym, "name", None)
                                if name:
                                    clean_name = name.split(":")[0]
                                    tensor_cache[clean_name] = val
                                    tensor_cache[name] = val
                            except Exception:
                                pass
                    else:
                        for out_idx, sym in enumerate(outputs):
                            tensor_cache[id(sym)] = current_tensor
                            try:
                                layer_key = ("layer_node_output", layer.name, node_idx, out_idx)
                                tensor_cache[layer_key] = current_tensor
                            except Exception:
                                pass
                            try:
                                name = getattr(sym, "name", None)
                                if name:
                                    clean_name = name.split(":")[0]
                                    tensor_cache[clean_name] = current_tensor
                                    tensor_cache[name] = current_tensor
                            except Exception:
                                pass

            final_outputs = []
            for symbolic_output in self.original_model.outputs:
                val = None
                try:
                    val = tensor_cache.get(id(symbolic_output), None)
                except Exception:
                    val = None

                if val is None:
                    try:
                        name = getattr(symbolic_output, "name", None)
                        if name:
                            clean_name = name.split(":")[0]
                            val = tensor_cache.get(clean_name, tensor_cache.get(name, None))
                    except Exception:
                        val = None

                if val is None:
                    try:
                        history = getattr(symbolic_output, "_keras_history", None)
                        if history and len(history) >= 3:
                            producing_layer_obj = history[0]
                            node_index = history[1]
                            tensor_index = history[2]
                            layer_key = ("layer_node_output", getattr(producing_layer_obj, "name", None), node_index, tensor_index)
                            val = tensor_cache.get(layer_key, None)
                    except Exception:
                        val = None

                if val is None:
                    try:
                        name = getattr(symbolic_output, "name", None)
                        if name:
                            base_name = name.split(":")[0].split("/")[-1]
                            val = tensor_cache.get(base_name, None)
                    except Exception:
                        val = None

                if val is None:
                    raise RuntimeError(
                        f"Missing runtime value for model output symbolic tensor: {symbolic_output}."
                        " Available tensor_cache keys: " + ",".join([str(k) for k in list(tensor_cache.keys())[:20]])
                    )

                final_outputs.append(val)

            if len(final_outputs) == 1:
                return final_outputs[0]
            return final_outputs

        def _apply_communication(self, sharded_output, layer_name, rule_str: str):
            """Applies communication directly using the distributed backend."""
            # This method is retained for backward compatibility if strings are passed.
            if "sum" in rule_str or "allreduce" in rule_str:
                logger.debug(f"Applying AllReduce (sum) to {layer_name}")
                return distribution_lib.all_reduce(
                    sharded_output, op="sum", axis_name="model"
                )

            elif "gather" in rule_str:
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