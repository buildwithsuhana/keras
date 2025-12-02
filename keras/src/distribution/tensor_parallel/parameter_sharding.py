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

    def _map_variables_to_owners(self, model):
        """
        Robustly maps variable object IDs to (layer, attribute_name).
        Uses dir() traversal to find variables hidden in lists or non-standard attributes.
        """
        var_to_owner = {}
        
        # Traverse all layers
        stack = [model]
        visited = set()
        
        while stack:
            layer = stack.pop()
            if id(layer) in visited:
                continue
            visited.add(id(layer))

            # 1. Scan all attributes (including hidden ones)
            for attr_name in dir(layer):
                if attr_name.startswith("__"): continue
                
                try:
                    attr_val = getattr(layer, attr_name, None)
                except Exception:
                    continue

                # Case A: Attribute is a Variable
                if isinstance(attr_val, keras.Variable):
                    var_to_owner[id(attr_val)] = (layer, attr_name)
                
                # Case B: Attribute is a Layer -> Add to stack
                elif hasattr(attr_val, "weights") and "Layer" in attr_val.__class__.__bases__[0].__name__:
                    stack.append(attr_val)
                
                # Case C: Attribute is a List/Tuple of Layers or Variables
                elif isinstance(attr_val, (list, tuple)):
                    for i, item in enumerate(attr_val):
                        if isinstance(item, keras.Variable):
                            # We can't easily replace inside a tuple/list via setattr
                            # but we track it. For lists, we might support it later if needed.
                            pass 
                        elif hasattr(item, "weights"):
                            stack.append(item)

            # 2. Standard Sub-layer traversal
            if hasattr(layer, 'layers'):
                stack.extend(layer.layers)

        return var_to_owner

    def _replace_variable(self, layer, attr_name, old_var, new_val_tensor):
        """
        Replaces 'old_var' with a new Variable containing 'new_val_tensor'.
        Bypasses Keras state locking using object.__setattr__.
        """
        # 1. Create new variable with correct shape
        new_var = keras.Variable(
            initializer=new_val_tensor,
            shape=new_val_tensor.shape,
            dtype=old_var.dtype,
            trainable=old_var.trainable,
            name=old_var.name
        )
        
        # 2. Hard Replace on Layer
        object.__setattr__(layer, attr_name, new_var)
        
        # 3. Update Keras internal tracking lists
        for lst_name in ['_trainable_weights', '_non_trainable_weights', '_weights', 'weights', 'variables']:
            if hasattr(layer, lst_name):
                lst = getattr(layer, lst_name)
                if isinstance(lst, list):
                    for i, v in enumerate(lst):
                        if v is old_var:
                            lst[i] = new_var
                            
        return new_var

    def shard_model_parameters(
        self,
        shard_model: "Model",
        weight_loader: Callable[[str], np.ndarray],
        config: Any,
        device_id: Any,
    ) -> Tuple["Model", Set[str]]:
        
        modified = set()
        var_to_owner = self._map_variables_to_owners(shard_model)
        
        for pattern, action in config.state_rules.items():
            if callable(action):
                targets = self._find_matching_parameters(shard_model, pattern)
                
                for name, target_var in targets:
                    try:
                        source_val = weight_loader(name)
                        if source_val is None: continue
                    except Exception:
                        continue

                    # Slice on CPU/RAM
                    sliced_val = action(source_val, self.rank)
                    
                    # Move to TPU
                    with keras.device(device_id):
                        sliced_val_tensor = ops.convert_to_tensor(sliced_val, dtype=target_var.dtype)
                    
                    # Assign or Resize
                    if target_var.shape == sliced_val_tensor.shape:
                        target_var.assign(sliced_val_tensor)
                    else:
                        if id(target_var) in var_to_owner:
                            layer, attr_name = var_to_owner[id(target_var)]
                            self._replace_variable(layer, attr_name, target_var, sliced_val_tensor)
                        else:
                            print(f"⚠️ Warning: Could not find owner for {name}. Attempting direct assign (Might Fail).")
                            target_var.assign(sliced_val_tensor)
                    
                    modified.add(name)
                    del source_val, sliced_val, sliced_val_tensor
        
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