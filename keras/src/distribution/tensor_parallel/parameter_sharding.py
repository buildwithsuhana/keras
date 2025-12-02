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
        Maps variable IDs to (layer, attribute_name).
        Prioritizes backing attributes (e.g. '_kernel') over properties (e.g. 'kernel').
        """
        # Local import to avoid circular dependency
        from keras.src.layers.layer import Layer
        
        print("\n🔍 [DEBUG] Starting Variable Owner Mapping...")
        var_to_owner = {}
        
        # Traverse all layers
        stack = [model]
        visited = set()
        
        while stack:
            layer = stack.pop()
            if id(layer) in visited:
                continue
            visited.add(id(layer))

            # 1. Scan attributes to find variables and sub-layers
            layer_vars = {} # id -> list of names
            
            for attr_name in dir(layer):
                if attr_name.startswith("__"): continue
                
                try:
                    attr_val = getattr(layer, attr_name, None)
                except Exception:
                    continue

                if isinstance(attr_val, keras.Variable):
                    layer_vars.setdefault(id(attr_val), []).append(attr_name)
                
                # Robustly check for nested layers
                elif isinstance(attr_val, Layer):
                    stack.append(attr_val)
                elif isinstance(attr_val, (list, tuple)):
                    for item in attr_val:
                        if isinstance(item, Layer):
                            stack.append(item)

            # 2. Resolve best attribute name for variables found on this layer
            for vid, names in layer_vars.items():
                # Default to first found
                best_name = names[0]
                for name in names:
                    # HEURISTIC: Prefer attributes starting with '_' (e.g. _kernel) 
                    # as these are usually the writable storage backing a property.
                    if name.startswith("_") and not best_name.startswith("_"):
                        best_name = name
                
                var_to_owner[vid] = (layer, best_name)
                # print(f"   -> Mapped var {vid} to {layer.name}.{best_name} (Candidates: {names})")

            # 3. Standard Sub-layer traversal
            if hasattr(layer, 'layers'):
                stack.extend(layer.layers)

        print(f"✅ [DEBUG] Mapped {len(var_to_owner)} variables to their owner layers.\n")
        return var_to_owner

    def _replace_variable(self, layer, attr_name, old_var, new_val_tensor):
        """
        Replaces 'old_var' with a new Variable containing 'new_val_tensor'.
        Uses object.__setattr__ to bypass built-layer locks and property setters.
        """
        print(f"      🛠️  [Resizing] Replacing '{attr_name}' on layer '{layer.name}'...")
        
        new_var = keras.Variable(
            initializer=new_val_tensor,
            shape=new_val_tensor.shape,
            dtype=old_var.dtype,
            trainable=old_var.trainable,
            name=old_var.name
        )
        
        # Force set the attribute (bypassing @property setters and Keras tracking)
        object.__setattr__(layer, attr_name, new_var)
        
        # Update Keras internal tracking lists to point to the new variable
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
        # Pre-map all variables to their owners
        var_to_owner = self._map_variables_to_owners(shard_model)
        
        for pattern, action in config.state_rules.items():
            if callable(action):
                targets = self._find_matching_parameters(shard_model, pattern)
                
                for name, target_var in targets:
                    print(f"⚡ [Sharding] Processing: {name}")
                    try:
                        source_val = weight_loader(name)
                        if source_val is None: 
                            print(f"   ⚠️ Source value not found for {name}, skipping.")
                            continue
                    except Exception as e:
                        print(f"   ⚠️ Error loading {name}: {e}")
                        continue

                    # 1. Slice on CPU (NumPy)
                    sliced_val = action(source_val, self.rank)
                    print(f"   -> Sliced {source_val.shape} -> {sliced_val.shape} (Rank {self.rank})")

                    # 2. Move to Device
                    with keras.device(device_id):
                        sliced_val_tensor = ops.convert_to_tensor(sliced_val, dtype=target_var.dtype)
                    
                    # 3. Assign or Resize
                    if target_var.shape == sliced_val_tensor.shape:
                        print(f"   -> Shapes match {target_var.shape}. Assigning directly.")
                        target_var.assign(sliced_val_tensor)
                    else:
                        print(f"   -> Shape Mismatch! Target: {target_var.shape} vs Sliced: {sliced_val_tensor.shape}")
                        
                        if id(target_var) in var_to_owner:
                            layer, attr_name = var_to_owner[id(target_var)]
                            self._replace_variable(layer, attr_name, target_var, sliced_val_tensor)
                        else:
                            # If mapping fails, we can't resize.
                            print(f"   ❌ CRITICAL WARNING: Could not find owner layer for {name}. Cannot resize variable!")
                            print(f"   ❌ Attempting direct assign (This will likely fail with ValueError)...")
                            target_var.assign(sliced_val_tensor)
                    
                    modified.add(name)
                    # Immediate cleanup to save RAM
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