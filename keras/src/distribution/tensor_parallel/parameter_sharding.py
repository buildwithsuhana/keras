import logging
import re

from keras import Variable
from keras import device
from keras.src import backend
from keras.src import layers
from keras.src.backend import distribution_lib as backend_dist_lib

logger = logging.getLogger(__name__)


class ParameterShardingStrategy:
    """Handles parameter-level sharding logic using native backend tensors."""

    def __init__(self, device_count, rank, device_mesh=None):
        self.device_count = device_count
        self.rank = rank
        self.device_mesh = device_mesh

    def shard_model_parameters(self, model, config, device_id):
        """Shards model parameters in-place using backend-native distribution."""
        print(f"🔧 Applying AUTOMATIC parameter-level sharding to {model.name}")

        from keras.src.distribution import distribution_lib as ag_dist_lib
        from keras.src.distribution.distribution_lib import TensorLayout
        
        # Priority: 1. explicit mesh from init, 2. global distribution, 3. model distribution
        device_mesh = self.device_mesh
        if device_mesh is None:
            dist_strategy = ag_dist_lib.distribution()
            device_mesh = dist_strategy.device_mesh if dist_strategy else None
        
        if device_mesh is None:
            device_mesh = getattr(model, "_distribution", None)
            if device_mesh:
                 device_mesh = device_mesh.device_mesh

        # 1. Map patterns/IDs to actions
        path_to_action = {}
        for pattern, action in config.state_rules.items():
            if callable(action):
                targets = self._find_matching_parameters(model, pattern)
                for name, param in targets:
                    path_to_action[name] = (param, action)

        # 2. Iterate through ALL weights to log status per device
        sharded_count = 0
        replicated_count = 0
        
        for w in model.weights:
            name = w.path
            if name in path_to_action:
                param, action = path_to_action[name]
                shard_dim = getattr(action, "keywords", {}).get("dim", 0)
                
                # Create layout for native distribution
                axes = [None] * len(param.shape)
                axes[shard_dim] = "model"
                layout = TensorLayout(tuple(axes), device_mesh=device_mesh) if device_mesh else None
                
                print(f"   [DEV: {device_id}] 🌐 Sharding {name} (NATIVE) along dim {shard_dim}")
                backend_dist_lib.distribute_variable(param, layout)
                sharded_count += 1
            else:
                # Replicated
                print(f"   [DEV: {device_id}] 🧬 Replicating {name}")
                replicated_count += 1

        # 3. Apply Layer Patching for Accuracy
        # This ensures bias is added AFTER AllReduce for RowParallel layers, 
        # and AllGather happens for ColumnParallel when needed.
        for layer in model._flatten_layers(recursive=True):
            rule = config.output_rules.get(layer.path)
            
            needs_patching = False
            add_bias_after = False
            
            # Row-Parallel Logic
            is_reduce_sum = (hasattr(rule, "rule_type") and rule.rule_type == "reduce_sum")
            if is_reduce_sum:
                needs_patching = True
                if hasattr(layer, "use_bias") and layer.use_bias:
                     # Disable bias in the shard. We add it after AllReduce in the patch.
                     layer.use_bias = False
                     add_bias_after = True
                     print(f"   [DEV: {device_id}] 🛠️  Patching RowParallel layer {layer.path} (bias moved after reduction)")
            
            # Column-Parallel Logic (Optional patching for explicit gather)
            is_gather = (hasattr(rule, "rule_type") and rule.rule_type == "gather")
            if is_gather:
                 needs_patching = True
                 print(f"   [DEV: {device_id}] 🛠️  Patching ColumnParallel layer {layer.path} (explicit gather)")

            if needs_patching:
                self._patch_layer(layer, rule, add_bias_after)

        # 4. Handle Dropout sharding (Parallel Regions)
        for layer in model._flatten_layers(recursive=True):
            if isinstance(layer, layers.Dropout) and config.output_rules.get(layer.path) == "parallel_dropout":
                if hasattr(layer, "seed_generator"):
                    # Offset the seed by rank to get different masks
                    from keras.src import ops
                    current_seed = ops.convert_to_numpy(layer.seed_generator.state)
                    new_seed = current_seed.copy()
                    new_seed[0] += self.rank
                    layer.seed_generator.state.assign(new_seed)
                    print(f"   [DEV: {device_id}] 🎲 Offset dropout seed for {layer.path} by {self.rank}")

        print(f"🎯 Rank {self.rank} ({device_id}) setup complete: {sharded_count} sharded, {replicated_count} replicated parameters")
        return model, set(path_to_action.keys())

    def _patch_layer(self, layer, output_rule, add_bias_after):
        """Intercepts layer.call to apply distribution rules and bias logic."""
        if hasattr(layer, "_is_tp_patched") and layer._is_tp_patched:
            return
        
        original_call = layer.call
        # We capture the bias variable before it's potentially modified
        bias = getattr(layer, "bias", None)
        
        def tp_call(inputs, *args, **kwargs):
            outputs = original_call(inputs, *args, **kwargs)
            if output_rule:
                outputs = output_rule(outputs)
            if add_bias_after and bias is not None:
                outputs = outputs + bias
            return outputs
        
        layer.call = tp_call
        layer._is_tp_patched = True

    def _find_matching_parameters(self, model, pattern):
        """Finds matching parameters by string pattern or object ID."""
        matches = []
        if isinstance(pattern, int):
            # Match by memory ID
            for w in model.weights:
                if id(w) == pattern:
                    return [(w.path, w)]
            return []

        # Match by string pattern
        for w in model.weights:
            if w.path == pattern or w.path.endswith("/" + pattern):
                matches.append((w.path, w))
        return matches


def make_parameter_sharded_model(module, config, rank, device_count, device_id, device_mesh=None):
    strat = ParameterShardingStrategy(device_count, rank, device_mesh=device_mesh)
    with device(device_id):
        return strat.shard_model_parameters(module, config, device_id)

    def _find_matching_parameters(self, model, pattern):
        """Finds matching parameters by string pattern or object ID."""
        matches = []
        if isinstance(pattern, int):
            # Match by memory ID
            for w in model.weights:
                if id(w) == pattern:
                    return [(w.path, w)]
            return []

        # Match by string pattern
        for w in model.weights:
            if w.path == pattern or w.path.endswith("/" + pattern):
                matches.append((w.path, w))
        return matches


def make_parameter_sharded_model(module, config, rank, device_count, device_id, device_mesh=None):
    strat = ParameterShardingStrategy(device_count, rank, device_mesh=device_mesh)
    with device(device_id):
        return strat.shard_model_parameters(module, config, device_id)
