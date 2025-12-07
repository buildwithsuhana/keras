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
        Traverses attributes, lists, and DICTS to find all layers and variables.
        """
        var_to_owner = {}
        
        stack = [model]
        visited = set()
        
        while stack:
            layer = stack.pop()
            if id(layer) in visited:
                continue
            visited.add(id(layer))

            layer_vars = {} 
            
            # Scan all attributes of the object
            for attr_name in dir(layer):
                if attr_name.startswith("__"): continue
                
                try:
                    attr_val = getattr(layer, attr_name, None)
                except Exception:
                    continue
                
                if attr_val is None: continue
                if isinstance(attr_val, type): continue

                # Duck typing checks
                is_var = hasattr(attr_val, 'assign') and hasattr(attr_val, 'value') and hasattr(attr_val, 'dtype')
                is_layer = (hasattr(attr_val, 'weights') and hasattr(attr_val, 'add_weight') and not is_var)

                if is_var:
                    layer_vars.setdefault(id(attr_val), []).append(attr_name)
                
                elif is_layer:
                    stack.append(attr_val)
                
                elif isinstance(attr_val, (list, tuple)):
                    for item in attr_val:
                        if item is None or isinstance(item, type): continue
                        if hasattr(item, 'weights') and hasattr(item, 'add_weight'):
                            stack.append(item)
                            
                # Dictionary Support
                elif isinstance(attr_val, dict):
                    for key, item in attr_val.items():
                        if item is None or isinstance(item, type): continue
                        if hasattr(item, 'weights') and hasattr(item, 'add_weight'):
                            stack.append(item)

            # Resolve best name for variables
            for vid, names in layer_vars.items():
                best_name = names[0]
                for name in names:
                    if name.startswith("_") and not best_name.startswith("_"):
                        best_name = name
                var_to_owner[vid] = (layer, best_name)

            # Standard Sub-layer traversal
            if hasattr(layer, 'layers'):
                try:
                    sublayers = layer.layers
                    if sublayers and hasattr(sublayers, '__iter__'):
                        stack.extend(sublayers)
                except Exception: pass

        return var_to_owner

    def _replace_variable(self, layer, attr_name, old_var, new_val_tensor):
        print(f"      🛠️  [Resizing] Replacing '{attr_name}' on layer '{layer.name}' (ID: {id(layer)})...")
        VarClass = old_var.__class__
        try:
            new_var = VarClass(
                initializer=new_val_tensor,
                shape=new_val_tensor.shape,
                dtype=old_var.dtype,
                trainable=old_var.trainable,
                name=old_var.name
            )
        except Exception:
            new_var = keras.Variable(
                initializer=new_val_tensor,
                shape=new_val_tensor.shape,
                dtype=old_var.dtype,
                trainable=old_var.trainable,
                name=old_var.name
            )
        
        # Bypass property setters and tracking locks
        try: 
            object.__setattr__(layer, attr_name, new_var)
        except Exception as e: 
            print(f"      ⚠️ object.__setattr__ failed: {e}. Trying setattr...")
            setattr(layer, attr_name, new_var)
        
        # Update Keras internal tracking lists
        # Added 'trainable_weights' and 'non_trainable_weights' explicitly just in case
        for lst_name in ['_trainable_weights', '_non_trainable_weights', '_weights', 'weights', 'variables', 'trainable_weights', 'non_trainable_weights']:
            if hasattr(layer, lst_name):
                lst = getattr(layer, lst_name)
                if isinstance(lst, list):
                    for i, v in enumerate(lst):
                        if v is old_var: lst[i] = new_var
        return new_var

    def _find_layer_by_path(self, model, var_path):
        """
        Traverses the model using the variable path to find the owner layer.
        """
        print(f"      🔎 [PathLookup] Starting lookup for: {var_path}")
        if ":" in var_path: var_path = var_path.split(":")[0]
        parts = var_path.split("/")
        
        attr_name = parts[-1]
        layer_path = parts[:-1]
        
        current = model
        
        def find_child(parent, name):
            # 1. Check direct attributes by checking .name property of attribute value
            for attr in dir(parent):
                if attr.startswith("__"): continue
                try: val = getattr(parent, attr)
                except: continue
                if hasattr(val, 'name') and val.name == name and hasattr(val, 'weights'):
                    return val
            
            # 2. Check layers list explicitly
            if hasattr(parent, 'layers'):
                for layer in parent.layers:
                    if layer.name == name:
                        return layer
            return None

        for part in layer_path:
            # 1. Try finding in current node
            if hasattr(current, 'name') and current.name == part:
                continue 
                
            next_node = find_child(current, part)
            
            if next_node:
                current = next_node
            else:
                # 2. Heuristic: Check for implicit wrappers
                print(f"        ⚠️ '{part}' not found in '{current.name}'. Checking sub-modules...")
                found_in_wrapper = False
                for wrapper in ['backbone', 'model', 'encoder', 'decoder', 'transformer', 'preprocessor']:
                    if hasattr(current, wrapper):
                        sub_module = getattr(current, wrapper)
                        candidate = find_child(sub_module, part)
                        if candidate:
                            print(f"        -> Found '{part}' inside '{wrapper}'. Diving in.")
                            current = candidate
                            found_in_wrapper = True
                            break
                
                if not found_in_wrapper:
                    print(f"        ❌ Failed to resolve path part '{part}'.")
                    return None, None

        # Check attribute - PREFER PRIVATE STORAGE (_attr) over properties (attr)
        if hasattr(current, "_" + attr_name):
            print(f"        -> Found '_{attr_name}' (preferred over '{attr_name}').")
            return current, "_" + attr_name
            
        if hasattr(current, attr_name):
            return current, attr_name
            
        print(f"        ❌ Attribute '{attr_name}' not found on final layer '{current.name}'.")
        return None, None

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
                    print(f"⚡ [Sharding] Processing: {name}")
                    try:
                        source_val = weight_loader(name)
                        if source_val is None: 
                            print(f"   ⚠️ Source value not found for {name}, skipping.")
                            continue
                    except Exception as e:
                        print(f"   ⚠️ Error loading {name}: {e}")
                        continue

                    sliced_val = action(source_val, self.rank)
                    
                    with keras.device(device_id):
                        sliced_val_tensor = ops.convert_to_tensor(sliced_val, dtype=target_var.dtype)
                    
                    if target_var.shape == sliced_val_tensor.shape:
                        target_var.assign(sliced_val_tensor)
                    else:
                        layer = None
                        attr_name = None
                        
                        # Strategy A: Use ID Mapping
                        if id(target_var) in var_to_owner:
                            layer, attr_name = var_to_owner[id(target_var)]
                        
                        # Strategy B: Path-based Fallback
                        if layer is None:
                            var_path = target_var.path if hasattr(target_var, 'path') else target_var.name
                            print(f"   ⚠️ [Fallback] ID mapping failed. Attempting path lookup for: {var_path}")
                            layer, attr_name = self._find_layer_by_path(shard_model, var_path)

                        if layer and attr_name:
                            self._replace_variable(layer, attr_name, target_var, sliced_val_tensor)
                        else:
                            print(f"   ❌ CRITICAL WARNING: Could not find owner layer for {name}. Cannot resize variable!")
                            print(f"   ❌ Attempting direct assign (This will likely fail with ValueError)...")
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
