import logging
import inspect
import re
import functools

import numpy as np

from keras.src.backend import distribution_lib
from keras.src import ops
from keras.src import layers
from keras import Variable, device 

from keras.src.distribution.tensor_parallel.tensor_layout import LayoutMap, split_tensor_for_parallelism

logger = logging.getLogger(__name__)


class ShardedWeight:
    """A wrapper for a sharded Keras Variable providing a consistent interface.

    Attributes:
        regularizer: Placeholder for weight regularization logic.
    """

    def __init__(self, tensor_shard, name, trainable=True, device_id=None):
        """Initializes the ShardedWeight.

        Args:
            tensor_shard: The actual tensor slice for this specific rank.
            name: Original name of the weight.
            trainable: Whether the weight is trainable.
            device_id: The specific device identifier (e.g., 'cpu:0').
        """
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
        """Returns the name of the wrapped variable."""
        return self._variable.name

    @property
    def trainable(self):
        """Returns the trainability status of the wrapped variable."""
        return self._variable.trainable

    @property
    def shape(self):
        """Returns the shape of the sharded tensor."""
        return self._variable.shape

    @property
    def dtype(self):
        """Returns the data type of the wrapped variable."""
        return self._variable.dtype

    @property
    def variable(self):
        """Returns the underlying Keras Variable."""
        return self._variable

    @property
    def value(self):
        """Returns the value of the underlying variable."""
        return self._variable.value

    def numpy(self):
        """Returns the numpy representation of the weight shard."""
        return self._variable.numpy()

    def num_elements(self):
        """Returns the total number of elements in this shard."""
        return ops.size(self._variable)

    def __repr__(self):
        return (
            f"<ShardedWeight name='{self.name}' "
            f"shape={self.shape} trainable={self.trainable}>"
        )


class ParameterShardingStrategy:
    """Strategy for sharding model parameters across multiple devices.

    This class handles the logic of identifying which parameters should be sharded,
    performing the split operations, and normalizing configuration keys.
    """

    def __init__(self, device_count, rank):
        """Initializes the sharding strategy.

        Args:
            device_count: Total number of devices in the model mesh.
            rank: The specific rank of the current device.
        """
        self.device_count = device_count
        self.rank = rank
        self.sharded_weights = {}
        self.original_weights = {}
        self.weight_mapping = {}
        self.sharded_weights_by_id = {}
        self.param_path_map = {}
        self._id_to_param_map = {}

    def shard_model_parameters(self, model, config, device_id):
        """Shards model parameters based on the provided configuration.

        Args:
            model: The original Keras model to shard.
            config: A LayoutMap containing state and output rules.
            device_id: The specific device for this shard.

        Returns:
            A tuple of (sharded_model, modified_parameters_set).
        """
        ParameterShardedModel = _define_parameter_sharded_model()

        print(f"🔧 Applying parameter-level sharding to {model.name}")

        self.param_path_map = {w.path: w for w in model.weights}
        
        for w in model.weights:
            self._id_to_param_map[id(w)] = (w.path, w)
            exp_ref_fn = getattr(w, "experimental_ref", None)
            if exp_ref_fn:
                self._id_to_param_map[id(exp_ref_fn())] = (w.path, w)
        
        # Discover missing embeddings like PositionEmbedding to ensure full sharding (122 params).
        for layer in model._flatten_layers(recursive=True, include_self=True):
            if "Embedding" in layer.__class__.__name__:
                for attr in ["embeddings", "position_embeddings", "_embeddings"]:
                    var = getattr(layer, attr, None)
                    if var is not None:
                        var_ref = getattr(var, "experimental_ref", lambda: var)()
                        var_id = id(var_ref)
                        if var_id not in config.state_rules:
                            config.state_rules[var_id] = functools.partial(
                                split_tensor_for_parallelism, 
                                device_count=self.device_count, 
                                dim=1
                            )

        # Normalize config by mapping integer IDs back to string paths for library compatibility.
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
        """Caches original weights before sharding."""
        for weight in model.weights:
            if hasattr(weight, 'numpy'):
                self.original_weights[weight.path] = weight.numpy()

    def _find_matching_parameters(self, model, pattern):
        """Finds model parameters that match a specific pattern or ID.

        Args:
            model: The model to search.
            pattern: A string path, suffix, or integer ID.

        Returns:
            A list of (path, weight) tuples.
        """
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
    """Factory to define the ParameterShardedModel class to break circular imports."""
    from keras.src.models import Model, Functional

    class ParameterShardedModel(Model):
        """Wrapper model that executes the forward pass with sharded parameters."""

        def __init__(self, original_model, sharding_strategy, config, device_id):
            """Initializes the wrapper model.

            Args:
                original_model: The Keras model being distributed.
                sharding_strategy: The strategy instance containing shard data.
                config: The LayoutMap for the model.
                device_id: Current device ID.
            """
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
            """Returns the current device of this model shard."""
            return self._device

        def _build_and_cache_weights(self):
            """Constructs the definitive list of sharded and unsharded weights."""
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
            """Returns the full list of weights (sharded and unsharded)."""
            return self._weights_list
        
        def compute_output_shape(self, input_shape):
            """Forwards output shape computation to the original model."""
            return self.original_model.compute_output_shape(input_shape)

        def compute_output_spec(self, *args, **kwargs):
            """Forwards output spec computation to the original model."""
            if args:
                return self.original_model.compute_output_spec(args[0])
            return self.original_model.compute_output_spec(**kwargs)

        def call(self, inputs, training=None, mask=None):
            """Performs the forward pass using sharded weights and communication.

            Args:
                inputs: Top-level model inputs.
                training: Boolean for training mode.
                mask: Optional input mask.

            Returns:
                Model outputs after applying communication rules.
            """
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
                
                reconstructed_inputs = []
                for node in layer._inbound_nodes:
                    for symbolic_input_tensor in node.input_tensors:
                        val = tensor_cache.get(id(symbolic_input_tensor))
                        if val is None:
                            name = getattr(symbolic_input_tensor, "name", None)
                            if name:
                                val = tensor_cache.get(name.split(":")[0], tensor_cache.get(name))
                        if val is not None:
                            reconstructed_inputs.append(val)
                
                # Reconstruct inputs based on layer expectations.
                layer_inputs = None
                if len(reconstructed_inputs) == 0:
                    layer_inputs = inputs
                else:
                    if isinstance(layer, (Functional, Model)):
                        layer_input_names = getattr(layer, "input_names", [])
                        if not layer_input_names and hasattr(layer, "inputs") and layer.inputs:
                            layer_input_names = [getattr(x, "name", "").split(":")[0] for x in layer.inputs]
                        
                        if layer_input_names and len(reconstructed_inputs) == len(layer_input_names):
                            layer_inputs = dict(zip(layer_input_names, reconstructed_inputs))
                        else:
                            layer_inputs = reconstructed_inputs
                    elif len(reconstructed_inputs) == 1:
                        layer_inputs = reconstructed_inputs[0]
                    else:
                        layer_inputs = reconstructed_inputs

                call_kwargs = {"training": training} if training is not None else {}
                node_args = getattr(node, "arguments", None)
                if node_args:
                    extra_kwargs = getattr(node_args, "kwargs", {})
                    if extra_kwargs:
                        for k, v in extra_kwargs.items():
                            if k != "training":
                                call_kwargs[k] = v

                current_tensor = layer(layer_inputs, **call_kwargs)

                # 4. Apply Communication Rules (ReduceSum / AllGather)
                layer_path = getattr(layer, "path", layer.name)
                output_rule = None
                for pattern, rule in self.config.output_rules.items():
                    if isinstance(pattern, str):
                        if pattern == layer_path or layer_path.endswith("/" + pattern) or re.search(pattern, layer_path):
                            output_rule = rule.get(0) if isinstance(rule, dict) else rule
                            break

                if output_rule:
                    if callable(output_rule):
                        current_tensor = output_rule(current_tensor)
                    else:
                        current_tensor = self._apply_communication(current_tensor, layer.name, output_rule)

                # 5. Map Runtime Outputs to Symbolic Tensors in Cache
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

        def _apply_communication(self, sharded_output, layer_name, rule_str):
            """Calls backend distribution functions based on rule strings."""
            if "sum" in rule_str or "allreduce" in rule_str:
                return distribution_lib.all_reduce(sharded_output, op="sum", axis_name="model")
            if "gather" in rule_str:
                parts = rule_str.split(" ")
                dim = int(parts[-1]) if len(parts) > 1 and parts[-1].lstrip('-').isdigit() else -1
                return distribution_lib.all_gather(sharded_output, axis=dim, axis_name="model")
            return sharded_output

        def get_config(self):
            """Returns the configuration of the original model."""
            return self.original_model.get_config()

        @classmethod
        def from_config(cls, config, custom_objects=None):
            """Rebuilds the sharded model from config."""
            return cls(**config)

    return ParameterShardedModel


def make_parameter_sharded_model(module, config, rank, device_count, device_id):
    """Factory function to create a parameter-sharded model.

    Args:
        module: Original model instance.
        config: LayoutMap.
        rank: Device rank.
        device_count: Total devices.
        device_id: Device identifier.

    Returns:
        A tuple of (sharded_model, modified_parameters_set).
    """
    sharding_strategy = ParameterShardingStrategy(device_count, rank)
    return sharding_strategy.shard_model_parameters(module, config, device_id)