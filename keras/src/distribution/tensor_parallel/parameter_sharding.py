import logging
import re
from typing import Any, Tuple, Set, TYPE_CHECKING
import gc

from keras import Variable, device
from keras import ops
from keras.src import layers

if TYPE_CHECKING:
    from keras.src.models import Model

logger = logging.getLogger(__name__)

class ParameterShardingStrategy:
    def __init__(self, device_count: int, rank: int):
        self.device_count = device_count
        self.rank = rank

    def shard_model_parameters(
        self,
        shard_model: "Model",
        source_model: "Model",
        config: Any,
        device_id: Any,
    ) -> Tuple["Model", Set[str]]:
        """
        Shards the `shard_model` in-place using weights from `source_model`.
        Crucially, this does NOT create a copy of all weights on CPU.
        """
        print(f"   [Rank {self.rank}] 🔧 Sharding weights onto {device_id}...")
        
        modified_parameters = set()
        
        # We need a quick lookup for source variables to avoid repeated iteration
        # Mapping: Clean Name -> Variable Object
        source_vars_map = {v.path if hasattr(v, 'path') else v.name: v for v in source_model.variables}
        
        # Iterate over the rules and apply them
        for pattern, action in config.state_rules.items():
            if callable(action):
                # Find parameters in the SHARD model to replace
                target_params = self._find_matching_parameters(shard_model, pattern)

                for param_name, target_param in target_params:
                    if param_name not in source_vars_map:
                        # Fallback: try finding by suffix if exact path mismatch happens
                        # (Common in some Keras clone_model edge cases)
                        suffix = param_name.split('/')[-1]
                        candidates = [v for k, v in source_vars_map.items() if k.endswith(suffix)]
                        if len(candidates) == 1:
                            source_param = candidates[0]
                        else:
                            continue
                    else:
                        source_param = source_vars_map[param_name]
                    
                    # EXECUTE SHARDING ACTION
                    # 1. Slice the master weight (CPU) -> creates a smaller slice
                    try:
                        sharded_value = action(source_param, self.rank)
                    except Exception as e:
                        logger.warning(f"Failed to shard {param_name}: {e}")
                        continue
                    
                    # 2. Assign to the target variable on the specific device
                    # We use 'assign' to update the existing variable in the cloned model
                    target_param.assign(sharded_value)
                        
                    modified_parameters.add(param_name)
        
        # Force cleanup of any intermediate slicing tensors
        gc.collect()
        return shard_model, modified_parameters

    def _find_matching_parameters(self, model, pattern: str):
        """Helper to find parameters matching a regex."""
        matches = []
        for v in model.variables:
            name = v.path if hasattr(v, 'path') else v.name
            if re.search(pattern, name):
                matches.append((name, v))
        return matches

def make_parameter_sharded_model(
    shard_model: "Model",
    source_model: "Model",
    config: Any,
    rank: int,
    device_count: int,
    device_id: Any,
) -> Tuple["Model", Set[str]]:
    """
    Apply sharding to a clone of the model.
    """
    strategy = ParameterShardingStrategy(device_count, rank)
    return strategy.shard_model_parameters(shard_model, source_model, config, device_id)