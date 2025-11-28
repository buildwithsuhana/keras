import logging
import re
import gc
from typing import Any, Tuple, Set, TYPE_CHECKING
from keras import device

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
        
        modified = set()
        source_map = {v.path if hasattr(v,'path') else v.name: v for v in source_model.variables}

        for pattern, action in config.state_rules.items():
            if callable(action):
                targets = self._find_matching_parameters(shard_model, pattern)
                for name, target_var in targets:
                    if name not in source_map: continue
                    source_var = source_map[name]
                    sliced_val = action(source_var, self.rank)
                    target_var.assign(sliced_val)
                    modified.add(name)
        
        gc.collect()
        return shard_model, modified

    def _find_matching_parameters(self, model, pattern: str):
        matches = []
        for v in model.variables:
            name = v.path if hasattr(v, 'path') else v.name
            if re.search(pattern, name):
                matches.append((name, v))
        return matches

def make_parameter_sharded_model(shard_model, source_model, config, rank, device_count, device_id):
    strategy = ParameterShardingStrategy(device_count, rank)
    return strategy.shard_model_parameters(shard_model, source_model, config, device_id)