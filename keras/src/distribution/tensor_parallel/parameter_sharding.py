import logging
import re
import gc
from typing import Any, Tuple, Set, Callable, TYPE_CHECKING
from keras import device, ops
import keras
import numpy as np

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
        weight_loader: Callable[[str], np.ndarray],
        config: Any,
        device_id: Any,
    ) -> Tuple["Model", Set[str]]:
        
        modified = set()
        
        # Iterate rules
        for pattern, action in config.state_rules.items():
            if callable(action):
                # Find targets in the skeleton model
                targets = self._find_matching_parameters(shard_model, pattern)
                
                for name, target_var in targets:
                    # 1. Load Source Weight from Disk (Low Memory Impact)
                    try:
                        source_val = weight_loader(name)
                        if source_val is None:
                            logger.warning(f"Weight not found for {name}")
                            continue
                    except Exception as e:
                        logger.warning(f"Error loading {name}: {e}")
                        continue

                    # 2. Slice on CPU
                    # source_val is a numpy array here
                    sliced_val = action(source_val, self.rank)
                    
                    # 3. Explicitly Move Slice to Target GPU
                    with keras.device(device_id):
                        sliced_val = ops.convert_to_tensor(sliced_val)
                    
                    # 4. Assign & Update
                    # This replaces the large random weight in shard_model with the smaller sliced weight
                    target_var.assign(sliced_val)
                    
                    # Explicit cleanup
                    del source_val
                    del sliced_val
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

def make_parameter_sharded_model(shard_model, weight_loader, config, rank, device_count, device_id):
    strategy = ParameterShardingStrategy(device_count, rank)
    return strategy.shard_model_parameters(shard_model, weight_loader, config, device_id)