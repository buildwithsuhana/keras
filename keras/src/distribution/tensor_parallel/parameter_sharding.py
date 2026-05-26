import logging
import re
import functools

import numpy as np

from keras.src.backend import distribution_lib
from keras.src import backend
from keras.src import ops
from keras.src import layers
from keras import Variable, device 

from keras.src.distribution.tensor_parallel.tensor_layout import LayoutMap, split_tensor_for_parallelism

logger = logging.getLogger(__name__)


class ShardedWeight:
    """Wrapper for a sharded Keras Variable providing a consistent interface.

    Attributes:
        regularizer: Placeholder for weight regularization logic.
    """

    def __init__(self, tensor_shard, name, trainable=True, device_id=None):
        """Initializes the ShardedWeight.

        Args:
            tensor_shard: The tensor slice belonging to this rank.
            name: The original variable name.
            trainable: Boolean indicating if the weight is trainable.
            device_id: Device identifier string for the variable placement.
        """
        dev_name = device_id if device_id else "UNKNOWN_DEVICE"
        print(f"   [DEV: {dev_name}] 🧬 Creating Sharded Variable '{name}' shape {tensor_shard.shape}")

        safe_name = name.replace("/", "_").replace(":", "_")
        with device(dev_name):
            self._variable = Variable(initializer=tensor_shard, trainable=trainable, name=safe_name)
        
        # Ensure the variable has a path for optimizer tracking
        self._variable._path = name
        self.path = name
        self.regularizer = None

    @property
    def name(self):
        """Returns variable name."""
        return self._variable.name

    @property
    def trainable(self):
        """Returns trainability."""
        return self._variable.trainable

    @property
    def shape(self):
        """Returns sharded shape."""
        return self._variable.shape

    @property
    def dtype(self):
        """Returns data type."""
        return self._variable.dtype

    @property
    def variable(self):
        """Returns the internal Variable object."""
        return self._variable

    @property
    def value(self):
        """Returns the variable value."""
        return self._variable.value

    def numpy(self):
        """Returns numpy representation."""
        return self._variable.numpy()

    def num_elements(self):
        """Returns total element count."""
        return ops.size(self._variable)

    def __repr__(self):
        return f"<ShardedWeight name='{self.name}' shape={self.shape} trainable={self.trainable}>"


class ParameterShardingStrategy:
    """Handles parameter-level sharding logic and configuration normalization."""

    def __init__(self, device_count, rank):
        """Initializes the strategy.

        Args:
            device_count: Total devices in the model axis.
            rank: Current device rank.
        """
        self.device_count = device_count
        self.rank = rank
        self.sharded_weights = {}
        self.weight_mapping = {}
        self.sharded_weights_by_id = {}
        self.param_path_map = {}
        self._id_to_param_map = {}

    def shard_model_parameters(self, model, config, device_id):
        """Orchestrates the sharding of model parameters.

        Args:
            model: Original model instance.
            config: LayoutMap with sharding rules.
            device_id: Targeted device identifier.

        Returns:
            Tuple of (sharded_model, modified_parameters_set).
        """
        ParameterShardedModel = _define_parameter_sharded_model()
        print(f"🔧 Applying parameter-level sharding to {model.name}")

        # Map build & Auto-Discovery
        self.param_path_map = {w.path: w for w in model.weights}
        for w in model.weights:
            ref_fn = getattr(w, "experimental_ref", None)
            ref = ref_fn() if ref_fn else w
            self._id_to_param_map[id(ref)] = (w.path, w)

        # Preserve the original variable metadata to avoid mutating the model
        # across multiple shard creations for different ranks.
        original_metadata = {}
        for w in model.weights:
            original_metadata[id(w)] = (
                getattr(w, "_shape", None),
                getattr(w, "_ndim", None),
                getattr(w, "_value", None),
            )

        # Normalize configuration keys to string paths
        norm_rules = {}
        for pattern, action in list(config.state_rules.items()):
            if isinstance(pattern, int) and pattern in self._id_to_param_map:
                path, _ = self._id_to_param_map[pattern]
                norm_rules[path] = action
                del config.state_rules[pattern]
        config.state_rules.update(norm_rules)

        modified = set()

        # 1. Shard variables and apply shape-lying hack
        for pattern, action in config.state_rules.items():
            if callable(action):
                for name, param in self._find_matching_parameters(model, pattern):
                    if name in self.sharded_weights:
                        modified.add(name)
                        continue

                    pr_fn = getattr(param, "experimental_ref", None)
                    pid = id(pr_fn()) if pr_fn else id(param)

                    if pid in self.sharded_weights_by_id:
                        self.sharded_weights[name] = self.sharded_weights_by_id[pid]
                        modified.add(name)
                        continue

                    original_shape = param.shape
                    shard = action(param, self.rank)
                    self.sharded_weights[name] = shard
                    self.sharded_weights_by_id[pid] = shard
                    self.weight_mapping[name] = {"original": original_shape, "sharded": shard.shape}
                    
                    # Update original variable to sharded state (backend-agnostic hack)
                    shard_tensor = self._convert_to_tensor(shard)
                    param._shape = shard.shape
                    param._ndim = len(shard.shape)
                    if hasattr(param, "_value"):
                        param._value = shard_tensor
                    
                    modified.add(name)
                    print(f"   ✅ Sharded {name}: {original_shape} -> {shard.shape}")

        # 2. Patch layers recursively with output rules (communication ops)
        print("🔗 Patching layers with communication rules...")
        patched_count = 0
        for layer in model._flatten_layers(recursive=True, include_self=True):
            lp = getattr(layer, "path", None) or layer.name
            lp_s = str(lp)
            
            # Find and apply sharded variables to layer attributes
            for attr_name in dir(layer):
                try:
                    attr = getattr(layer, attr_name)
                    if isinstance(attr, Variable):
                        if attr.path in self.sharded_weights:
                             # Shape-lying for the layer variable itself
                             sharded_val = self.sharded_weights[attr.path]
                             sharded_tensor = self._convert_to_tensor(sharded_val)
                             attr._shape = sharded_val.shape
                             attr._ndim = len(attr._shape)
                             if hasattr(attr, "_value"):
                                 attr._value = sharded_tensor
                except Exception:
                    continue

            for pat, rule in config.output_rules.items():
                pat_s = str(pat)
                # Flexible matching: exact, suffix, or regex
                matched = (pat_s == lp_s or 
                           lp_s.endswith("/" + pat_s) or 
                           pat_s.endswith("/" + lp_s) or
                           (isinstance(pat_s, str) and re.search(pat_s, lp_s)))
                
                if matched:
                    actual_rule = rule.get(0) if isinstance(rule, dict) else rule
                    self._patch_layer(layer, actual_rule)
                    patched_count += 1
                    break
        print(f"🎯 Patched {patched_count} layers with communication rules")

        sharded_model = ParameterShardedModel(model, self, config, device_id)

        # Restore the original model variable metadata so subsequent shard
        # creations start from the full original model state.
        for w in model.weights:
            metadata = original_metadata.get(id(w))
            if metadata is not None:
                shape, ndim, value = metadata
                if shape is not None:
                    w._shape = shape
                if ndim is not None:
                    w._ndim = ndim
                if value is not None:
                    w._value = value

        print(f"🎯 Sharding complete: {len(modified)} parameters sharded")
        return sharded_model, modified

    def _patch_layer(self, layer, rule):
        """Patches a layer's call and compute_output_shape methods."""
        if hasattr(layer, "_is_tp_patched"):
            return
        
        # Patch call
        old_call = layer.call
        @functools.wraps(old_call)
        def sharded_call(*args, **kwargs):
            # Rule 3: For Row Parallel layers, bias should be added AFTER all_reduce
            from keras.src.distribution.tensor_parallel.autoconfig import _reduce_sum
            
            if rule == "parallel_dropout":
                # Rule 5: Parallel regions use different seeds
                if hasattr(layer, "seed_generator"):
                    # We slightly shift the seed based on rank to get different masks
                    # This is a simple way to achieve Rule 5 parallel RNG behavior
                    seed_state = layer.seed_generator.state
                    seed_state.assign(seed_state.value + self.rank * 1000)
                return old_call(*args, **kwargs)

            use_bias = getattr(layer, "use_bias", False)
            if use_bias and rule == _reduce_sum:
                # Temporarily disable bias addition in the original call
                layer.use_bias = False
                out = old_call(*args, **kwargs)
                layer.use_bias = True
                
                # Apply the rule (AllReduce)
                out = rule(out)
                
                # Add the bias manually after AllReduce
                out = out + layer.bias
                return out

            out = old_call(*args, **kwargs)
            if rule:
                if callable(rule):
                    out = rule(out)
                elif isinstance(rule, str):
                    out = self._comm(out, rule)
            return out
        layer.call = sharded_call

        # Disable input_spec validation for sharded layers
        # as it often conflicts with sharded input shapes.
        if hasattr(layer, "input_spec"):
            layer.input_spec = None

        # Patch compute_output_shape to return full shape if gathering
        old_cos = layer.compute_output_shape
        @functools.wraps(old_cos)
        def sharded_cos(input_shape):
            shape = old_cos(input_shape)
            if isinstance(rule, str) and "gather" in rule:
                parts = rule.split(" ")
                dim = int(parts[-1]) if len(parts) > 1 and parts[-1].lstrip('-').isdigit() else -1
                # Adjust negative axis to positive
                axis = dim if dim >= 0 else len(shape) + dim
                new_shape = list(shape)
                if new_shape[axis] is not None:
                    new_shape[axis] *= self.device_count
                return tuple(new_shape)
            return shape
        layer.compute_output_shape = sharded_cos
        
        layer._is_tp_patched = True
        print(f"   🔗 Patched layer {layer.name} with communication rule")

    def _get_layer_mlp_type(self, layer):
        from keras.src.distribution.tensor_parallel.autoconfig import analyze_dense_layer
        return analyze_dense_layer(layer)

    def _comm(self, val, rule):
        """Internal communication wrapper."""
        if rule == "parallel_dropout":
            # Rule 5: Parallel Dropout needs different seeds on different devices.
            # We achieve this by adding the rank to the seed if possible, 
            # or by doing nothing if the backend already handles it.
            # In Keras 3.0, we can use a seed_generator or just rely on the rank
            # being part of the global state.
            # For simplicity, we ensure that if we are in a parallel region, 
            # we are not syncing seeds.
            return val
        if "sum" in rule or "allreduce" in rule:
            res = distribution_lib.all_reduce(val, op="sum", axis_name="model")
            return res
        if "gather" in rule:
            parts = rule.split(" ")
            dim = int(parts[-1]) if len(parts) > 1 and parts[-1].lstrip('-').isdigit() else -1
            res = distribution_lib.all_gather(val, axis=dim, axis_name="model")
            return res
        return val

    def _convert_to_tensor(self, value):
        """Convert a value to a backend tensor with a stable fallback."""
        try:
            return ops.convert_to_tensor(value)
        except TypeError:
            return backend.core.convert_to_tensor(value)

    def _find_matching_parameters(self, model, pattern):
        """Matches a pattern to model weights."""
        if isinstance(pattern, int):
            return [self._id_to_param_map[pattern]] if pattern in self._id_to_param_map else []
        if not isinstance(pattern, str):
            return []
        if pattern in self.param_path_map:
            return [(pattern, self.param_path_map[pattern])]
        suffix = "/" + pattern
        return [(p, w) for p, w in self.param_path_map.items() if p.endswith(suffix)]


def _define_parameter_sharded_model():
    """Defines the wrapper model class dynamically."""
    from keras.src.models import Model

    class ParameterShardedModel(Model):
        """Wrapper model implementing distributed forward pass logic via weight injection."""

        def __init__(self, original_model, sharding_strategy, config, device_id):
            """Initializes the model and caches mappings."""
            super().__init__(name=original_model.name)
            self.original_model = original_model
            self.sharding_strategy = sharding_strategy
            self.config = config
            self._device = device_id
            
            if not self.original_model.built and self.original_model.inputs:
                self.original_model.build(self.original_model.inputs[0].shape)
            
            self._build_and_cache_weights()
            # Set built flag to True and full_build to skip redundant fit initialization
            self.built = True
            self._built = True
            print("🚀 ParameterShardedModel created and marked as BUILT")

        def _build_and_cache_weights(self):
            """Merges sharded and original weights into a definitive list."""
            ws, self._var_map = [], {}
            sharded_ids = set(self.sharding_strategy.sharded_weights_by_id.keys())

            for name, shard in self.sharding_strategy.sharded_weights.items():
                sharded_var = ShardedWeight(shard, name, device_id=self._device).variable
                ws.append(sharded_var)
                # Map original variable path and ref to our new sharded Variable object
                orig_var = self.sharding_strategy.param_path_map.get(name)
                if orig_var is not None:
                    self._var_map[orig_var.path] = sharded_var
                    if hasattr(orig_var, "experimental_ref"):
                        ref = orig_var.experimental_ref()
                        self._var_map[id(ref)] = sharded_var

            for w in self.original_model.weights:
                if w.path not in self._var_map:
                    # Explicitly move non-sharded weights to the target device
                    with device(self._device):
                        new_v = Variable(
                            initializer=w.value,
                            trainable=w.trainable,
                            name=w.name.replace("/", "_").replace(":", "_") + "_sharded",
                        )
                        new_v._path = w.path
                    ws.append(new_v)
                    self._var_map[w.path] = new_v
                    if hasattr(w, "experimental_ref"):
                        ref = w.experimental_ref()
                        self._var_map[id(ref)] = new_v
            
            self._weights_list = ws
            self._trainable_weights_list = [v for v in ws if v.trainable]
            self._non_trainable_weights_list = [v for v in ws if not v.trainable]

        @property
        def weights(self):
            """Returns model weights (shards for sharded params, original for others)."""
            return self._weights_list

        @property
        def trainable_weights(self):
            return self._trainable_weights_list

        @property
        def non_trainable_weights(self):
            return self._non_trainable_weights_list

        def compute_output_shape(self, input_shape):
            return self.original_model.compute_output_shape(input_shape)

        def compute_output_spec(self, *args, **kwargs):
            if args:
                return self.original_model.compute_output_spec(args[0])
            return self.original_model.compute_output_spec(**kwargs)

        def call(self, inputs, training=None, mask=None):
            """Forward pass that correctly handles sharded variable state."""
            # Since we've already updated the layers' internal state via shape-lying,
            # we can call the original model directly.
            # The sharded variables are stored in self._weights_list and tracked via
            # the framework through self.trainable_weights and self.non_trainable_weights.
            
            # Build a temporary mapping of original_model variables to their sharded values
            # for the duration of this call
            original_state = {}
            
            # Temporarily replace original_model's variables with our sharded versions
            for orig_var in self.original_model.trainable_variables:
                var_id = id(orig_var)
                original_state[var_id] = (
                    getattr(orig_var, "_shape", None),
                    getattr(orig_var, "_ndim", None),
                    getattr(orig_var, "_value", None),
                )
                sharded_var = self._var_map.get(
                    orig_var.path,
                    self._var_map.get(
                        id(orig_var.experimental_ref()) 
                        if hasattr(orig_var, "experimental_ref") else id(orig_var),
                        None
                    )
                )
                if sharded_var is not None:
                    orig_var._shape = sharded_var.shape
                    orig_var._ndim = len(sharded_var.shape) if sharded_var.shape else 0
                    if hasattr(orig_var, "_value"):
                        orig_var._value = sharded_var.value
                        
            for orig_var in self.original_model.non_trainable_variables:
                var_id = id(orig_var)
                if var_id not in original_state:  # Only store if not already stored
                    original_state[var_id] = (
                        getattr(orig_var, "_shape", None),
                        getattr(orig_var, "_ndim", None),
                        getattr(orig_var, "_value", None),
                    )
                sharded_var = self._var_map.get(
                    orig_var.path,
                    self._var_map.get(
                        id(orig_var.experimental_ref())
                        if hasattr(orig_var, "experimental_ref") else id(orig_var),
                        None
                    )
                )
                if sharded_var is not None:
                    orig_var._shape = sharded_var.shape
                    orig_var._ndim = len(sharded_var.shape) if sharded_var.shape else 0
                    if hasattr(orig_var, "_value"):
                        orig_var._value = sharded_var.value
            
            try:
                # Call the original model
                outputs = self.original_model.call(inputs, training=training, mask=mask)
            finally:
                # Restore original variable state
                for orig_var in (
                    list(self.original_model.trainable_variables) +
                    list(self.original_model.non_trainable_variables)
                ):
                    var_id = id(orig_var)
                    if var_id in original_state:
                        shape, ndim, value = original_state[var_id]
                        if shape is not None:
                            orig_var._shape = shape
                        if ndim is not None:
                            orig_var._ndim = ndim
                        if value is not None and hasattr(orig_var, "_value"):
                            orig_var._value = value
            
            return outputs

        def get_config(self):
            return self.original_model.get_config()

        @classmethod
        def from_config(cls, config, custom_objects=None):
            return cls(**config)

    return ParameterShardedModel


def make_parameter_sharded_model(module, config, rank, device_count, device_id):
    """Factory function to create a parameter-sharded model."""
    strat = ParameterShardingStrategy(device_count, rank)
    return strat.shard_model_parameters(module, config, device_id)
