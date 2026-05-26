import logging
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
                    pr_fn = getattr(param, "experimental_ref", None)
                    pid = id(pr_fn()) if pr_fn else id(param)

                    if pid in self.sharded_weights_by_id:
                        self.sharded_weights[name] = self.sharded_weights_by_id[pid]
                        modified.add(name)
                        continue

                    shard = action(param, self.rank)
                    self.sharded_weights[name] = shard
                    self.sharded_weights_by_id[pid] = shard
                    self.weight_mapping[name] = {"original": param.shape, "sharded": shard.shape}
                    
                    # Update original variable shape to pass StatelessScope validation
                    param._shape = shard.shape
                    param._ndim = len(shard.shape)
                    
                    modified.add(name)
                    print(f"   ✅ Sharded {name}: {param.shape} -> {shard.shape}")

        # 2. Patch layers recursively with output rules (communication ops)
        print("🔗 Patching layers with communication rules...")
        patched_count = 0
        for layer in model._flatten_layers(recursive=True, include_self=True):
            lp = getattr(layer, "path", None) or layer.name
            lp_s = str(lp)
            
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
        print(f"🎯 Sharding complete: {len(modified)} parameters sharded")
        return sharded_model, modified

    def _patch_layer(self, layer, rule):
        """Patches a layer's call method to apply communication rules."""
        if hasattr(layer, "_is_tp_patched"):
            return
        
        old_call = layer.call
        @functools.wraps(old_call)
        def sharded_call(*args, **kwargs):
            out = old_call(*args, **kwargs)
            if rule:
                print(f"DEBUG: Layer {layer.name} output shape BEFORE rule: {getattr(out, 'shape', 'N/A')}")
                if callable(rule):
                    out = rule(out)
                elif isinstance(rule, str):
                    out = self._comm(out, rule)
                print(f"DEBUG: Layer {layer.name} output shape AFTER rule: {getattr(out, 'shape', 'N/A')}")
            return out
        
        layer.call = sharded_call
        layer._is_tp_patched = True
        print(f"   🔗 Patched layer {layer.name} with communication rule")

    def _comm(self, val, rule):
        """Internal communication wrapper."""
        if "sum" in rule or "allreduce" in rule:
            res = distribution_lib.all_reduce(val, op="sum", axis_name="model")
            return res
        if "gather" in rule:
            parts = rule.split(" ")
            dim = int(parts[-1]) if len(parts) > 1 and parts[-1].lstrip('-').isdigit() else -1
            res = distribution_lib.all_gather(val, axis=dim, axis_name="model")
            return res
        return val

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
            self.built = True
            print("🚀 ParameterShardedModel created successfully")

        def _build_and_cache_weights(self):
            """Merges sharded and original weights into a definitive list."""
            ws, self._var_map = [], {}
            sharded_ids = set(self.sharding_strategy.sharded_weights_by_id.keys())
            
            for name, shard in self.sharding_strategy.sharded_weights.items():
                sharded_var = ShardedWeight(shard, name, device_id=self._device).variable
                ws.append(sharded_var)
                # Map original variable ID to our new sharded Variable object
                # Find the original variable for this name
                orig_var = self.sharding_strategy.param_path_map.get(name)
                if orig_var is not None:
                    ref = orig_var.experimental_ref() if hasattr(orig_var, "experimental_ref") else orig_var
                    self._var_map[id(ref)] = sharded_var

            for w in self.original_model.weights:
                rf_fn = getattr(w, "experimental_ref", None)
                ref = rf_fn() if rf_fn else w
                if id(ref) not in sharded_ids:
                    ws.append(w)
            
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
            """Forward pass using top-level stateless_call with weight injection."""
            # Prepare tensors/Variables for all trainable and non-trainable weights
            t_vars = []
            for v in self.original_model.trainable_variables:
                ref = v.experimental_ref() if hasattr(v, "experimental_ref") else v
                t_vars.append(self._var_map.get(id(ref), v))
                
            nt_vars = []
            for v in self.original_model.non_trainable_variables:
                ref = v.experimental_ref() if hasattr(v, "experimental_ref") else v
                nt_vars.append(self._var_map.get(id(ref), v))

            # Since we patched the internal layers of original_model and updated their variables' shapes,
            # we can just call stateless_call on the whole model.
            out_tuple = self.original_model.stateless_call(
                t_vars, nt_vars, inputs, training=training, mask=mask
            )
            
            # Update non-trainable state (RNGs, BN stats)
            if isinstance(out_tuple, tuple) and len(out_tuple) >= 2:
                out, updated_nt_vars = out_tuple[0], out_tuple[1]
                for v, nv in zip(nt_vars, updated_nt_vars):
                    if hasattr(v, "assign") and v is not nv:
                        v.assign(nv)
                return out
            return out_tuple

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