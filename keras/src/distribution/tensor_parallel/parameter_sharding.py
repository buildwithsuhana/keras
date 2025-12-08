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
        var_to_owner = {}
        stack = [model]
        visited = set()
        
        while stack:
            layer = stack.pop()
            if id(layer) in visited: continue
            visited.add(id(layer))

            layer_vars = {} 
            for attr_name in dir(layer):
                if attr_name.startswith("__"): continue
                try: attr_val = getattr(layer, attr_name, None)
                except Exception: continue
                if attr_val is None: continue

                is_var = hasattr(attr_val, 'assign') and hasattr(attr_val, 'value')
                is_layer = (hasattr(attr_val, 'weights') and hasattr(attr_val, 'add_weight') and not is_var)

                if is_var:
                    layer_vars.setdefault(id(attr_val), []).append(attr_name)
                elif is_layer:
                    stack.append(attr_val)
                elif isinstance(attr_val, (list, tuple)):
                    for item in attr_val:
                        if hasattr(item, 'weights'): stack.append(item)

            for vid, names in layer_vars.items():
                # Heuristic: prefer names that don't start with underscore if available, else take first
                best_name = names[0]
                for name in names:
                    if not name.startswith("_"):
                        best_name = name
                        break
                var_to_owner[vid] = (layer, best_name)

            if hasattr(layer, 'layers'):
                try: stack.extend(layer.layers)
                except Exception: pass
        return var_to_owner

    def _replace_variable(self, layer, attr_name, old_var, new_val_tensor):
        print(f"      🛠️  [Re-creating] '{attr_name}' on '{layer.name}' -> {new_val_tensor.device}")
        
        # 1. Create NEW variable. This ensures it adopts the device of 'new_val_tensor' (The GPU)
        # explicitly, because we are using the 'initializer' which is the tensor itself.
        try:
            # We use a lambda initializer to bypass some internal Keras checks that might complain
            # about device mismatch if we passed the tensor directly as a value.
            # But passing tensor as initializer usually works to force device.
            new_var = keras.Variable(
                initializer=new_val_tensor,
                shape=new_val_tensor.shape,
                dtype=old_var.dtype,
                trainable=old_var.trainable,
                name=old_var.name
            )
        except Exception as e:
            print(f"      ⚠️ Var creation failed: {e}")
            return old_var

        # 2. Force overwrite the attribute on the layer
        try: 
            object.__setattr__(layer, attr_name, new_var)
        except Exception: 
            setattr(layer, attr_name, new_var)
        
        # 3. Update Keras internal tracking lists to point to the new GPU variable
        for lst_name in ['_trainable_weights', '_non_trainable_weights', '_weights', 'weights', 'variables']:
            if hasattr(layer, lst_name):
                lst = getattr(layer, lst_name)
                if isinstance(lst, list):
                    for i, v in enumerate(lst):
                        if v is old_var: 
                            lst[i] = new_var

        return new_var

    def _find_layer_by_path(self, model, var_path):
        if ":" in var_path: var_path = var_path.split(":")[0]
        parts = var_path.split("/")
        attr_name = parts[-1]
        
        current = model
        
        # Helper to find attribute or layer
        def find_child(parent, name):
            # Check attributes first
            for attr in dir(parent):
                if attr.startswith("__"): continue
                try: val = getattr(parent, attr)
                except: continue
                if hasattr(val, 'name') and val.name == name: return val
            # Check layers list
            if hasattr(parent, 'layers'):
                for layer in parent.layers:
                    if layer.name == name: return layer
            return None

        # Navigate path
        for part in parts[:-1]:
            if hasattr(current, 'name') and current.name == part: continue 
            next_node = find_child(current, part)
            if next_node: 
                current = next_node
            else:
                # Fallback: check inside common wrappers
                found = False
                for wrapper in ['backbone', 'model', 'encoder', 'decoder', 'transformer']:
                    if hasattr(current, wrapper):
                        candidate = find_child(getattr(current, wrapper), part)
                        if candidate: 
                            current = candidate
                            found = True
                            break
                if not found: return None, None

        if hasattr(current, "_" + attr_name): return current, "_" + attr_name
        if hasattr(current, attr_name): return current, attr_name
        return None, None

    def shard_model_parameters(self, shard_model, weight_loader, config, device_id):
        modified = set()
        var_to_owner = self._map_variables_to_owners(shard_model)
        
        # Resolve JAX Target Device
        jax_target = None
        if keras.config.backend() == "jax":
            import jax
            try:
                # Parse "cuda:0" -> jax device object
                d_str = str(device_id)
                idx = int(d_str.split(":")[-1]) if ":" in d_str else 0
                
                # Try finding specific backend first
                try: jax_target = jax.devices('gpu')[idx]
                except: jax_target = jax.devices()[idx]
                
                print(f"      🎯 JAX Target: {jax_target}")
            except: pass

        for pattern, action in config.state_rules.items():
            if callable(action):
                targets = self._find_matching_parameters(shard_model, pattern)
                for name, target_var in targets:
                    print(f"⚡ [Sharding] Processing: {name}")
                    try:
                        source_val = weight_loader(name)
                        if source_val is None: continue
                    except: continue

                    sliced_val = action(source_val, self.rank)
                    
                    # --- FORCE PLACEMENT ---
                    if jax_target is not None:
                        import jax
                        # Push to GPU immediately
                        sliced_val_tensor = jax.device_put(sliced_val, jax_target)
                        sliced_val_tensor = sliced_val_tensor.astype(target_var.dtype)
                    else:
                        with keras.device(device_id):
                            sliced_val_tensor = ops.convert_to_tensor(sliced_val, dtype=target_var.dtype)

                    # --- ALWAYS REPLACE VARIABLE ---
                    # We do NOT use assign() because it might respect the old CPU placement.
                    # We find the owner layer and swap the variable object entirely.
                    
                    layer, attr_name = None, None
                    if id(target_var) in var_to_owner:
                        layer, attr_name = var_to_owner[id(target_var)]
                    
                    if layer is None:
                        # Fallback to path lookup
                        var_path = target_var.path if hasattr(target_var, 'path') else target_var.name
                        layer, attr_name = self._find_layer_by_path(shard_model, var_path)

                    if layer and attr_name:
                        self._replace_variable(layer, attr_name, target_var, sliced_val_tensor)
                    else:
                        print(f"   ❌ CRITICAL: Could not find owner for {name}. Forced assign (May OOM).")
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