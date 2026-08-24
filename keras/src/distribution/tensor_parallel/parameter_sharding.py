import logging
import re

from keras import Variable
from keras import device
from keras.src import backend
from keras.src.backend import distribution_lib as backend_dist_lib

logger = logging.getLogger(__name__)


class ParameterShardingStrategy:
    """Handles parameter-level sharding logic using native backend tensors."""

    def __init__(self, device_count, rank):
        self.device_count = device_count
        self.rank = rank

    def shard_model_parameters(self, model, config, device_id):
        """Shards model parameters in-place using backend-native distribution."""
        print(f"🔧 Applying AUTOMATIC parameter-level sharding to {model.name}")

        from keras.src.distribution import distribution_lib as ag_dist_lib
        from keras.src.distribution.distribution_lib import TensorLayout
        dist_strategy = ag_dist_lib.distribution()
        device_mesh = dist_strategy.device_mesh if dist_strategy else None

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
                layout = TensorLayout(device_mesh, tuple(axes)) if device_mesh else None
                
                print(f"   [DEV: {device_id}] 🌐 Sharding {name} (NATIVE) along dim {shard_dim}")
                backend_dist_lib.distribute_variable(param, layout)
                sharded_count += 1
            else:
                # Replicated
                print(f"   [DEV: {device_id}] 🧬 Replicating {name}")
                replicated_count += 1

        print(f"🎯 Rank {self.rank} ({device_id}) setup complete: {sharded_count} sharded, {replicated_count} replicated parameters")
        return model, set(path_to_action.keys())

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


def make_parameter_sharded_model(module, config, rank, device_count, device_id):
    strat = ParameterShardingStrategy(device_count, rank)
    with device(device_id):
        return strat.shard_model_parameters(module, config, device_id)
