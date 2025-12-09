import logging
import re
import gc
import os
import psutil
import subprocess
from typing import Any, Tuple, Set, Callable, TYPE_CHECKING
from keras import device, ops
import keras
import numpy as np

if TYPE_CHECKING:
    from keras.src.models import Model

logger = logging.getLogger(__name__)

# --- Helper for Memory Logging ---
def log_stats(stage=""):
    # CPU
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 ** 2)
    
    # GPU
    gpu_str = ""
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
            encoding='utf-8'
        )
        mems = [int(x) for x in result.strip().split('\n') if x.strip()]
        for i, m in enumerate(mems):
            gpu_str += f"G{i}:{m}M "
    except: pass

    print(f"   📊 [Stats] {stage} | RAM: {mem_mb:.0f}MB | {gpu_str}")
# ---------------------------------

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
                best_name = names[0]
                # Prefer private names (e.g. _kernel) to bypass property locks
                for name in names:
                    if name.startswith("_"):
                        best_name = name
                        break
                var_to_owner[vid] = (layer, best_name)

            if hasattr(layer, 'layers'):
                try: stack.extend(layer.layers)
                except Exception: pass
        return var_to_owner

    def _replace_variable(self, layer, attr_name, old_var, new_val_tensor, device_id=None):
        print(f"      🛠️  [Swapping] '{attr_name}' on '{layer.name}' -> {new_val_tensor.device}")
        
        try:
            # We assume new_val_tensor is already on the correct device (GPU)
            # Using it as initializer forces the Variable to adopt that device.
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
        
        # Force set the attribute, bypassing "built" checks
        success = False
        try:
            object.__setattr__(layer, attr_name, new_var)
            success = True
        except Exception: pass

        if not success and not attr_name.startswith("_"):
            try:
                object.__setattr__(layer, "_" + attr_name, new_var)
                success = True
            except Exception: pass
        
        if not success:
            print(f"      ⚠️ FAILED to swap attribute '{attr_name}'. Memory leak risk!")

        # Update internal Keras lists so the new variable is actually used
        for lst_name in ['_trainable_weights', '_non_trainable_weights', '_weights', 'weights', 'variables']:
            if hasattr(layer, lst_name):
                lst = getattr(layer, lst_name)
                if isinstance(lst, list):
                    for i, v in enumerate(lst):
                        if v is old_var: lst[i] = new_var

        return new_var

    def _find_layer_by_path(self, model, var_path):
        if ":" in var_path: var_path = var_path.split(":")[0]
        parts = var_path.split("/")
        attr_name = parts[-1]
        current = model
        
        def find_child(parent, name):
            for attr in dir(parent):
                if attr.startswith("__"): continue
                try: val = getattr(parent, attr)
                except: continue
                if hasattr(val, 'name') and val.name == name: return val
            if hasattr(parent, 'layers'):
                for layer in parent.layers:
                    if layer.name == name: return layer
            return None

        for part in parts[:-1]:
            if hasattr(current, 'name') and current.name == part: continue 
            next_node = find_child(current, part)
            if next_node: current = next_node
            else:
                found = False
                for wrapper in ['backbone', 'model', 'encoder', 'decoder', 'transformer', 'preprocessor']:
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
        
        jax_target = None
        if keras.config.backend() == "jax":
            import jax
            try:
                d_str = str(device_id)
                idx = int(d_str.split(":")[-1]) if ":" in d_str else 0
                try: jax_target = jax.devices('gpu')[idx]
                except: jax_target = jax.devices()[idx]
                print(f"      🎯 JAX Target: {jax_target}")
            except: pass

        for pattern, action in config.state_rules.items():
            if callable(action):
                targets = self._find_matching_parameters(shard_model, pattern)
                for name, target_var in targets:
                    print(f"⚡ [Sharding] Processing: {name}")
                    log_stats(f"Before {name}")  # <--- LOG BEFORE

                    try:
                        source_val = weight_loader(name)
                        if source_val is None: continue
                    except: continue

                    sliced_val = action(source_val, self.rank)
                    
                    if jax_target is not None:
                        import jax
                        # Push to GPU immediately
                        sliced_val_tensor = jax.device_put(sliced_val, jax_target)
                        sliced_val_tensor = sliced_val_tensor.astype(target_var.dtype)
                    else:
                        with keras.device(device_id):
                            sliced_val_tensor = ops.convert_to_tensor(sliced_val, dtype=target_var.dtype)

                    layer, attr_name = None, None
                    if id(target_var) in var_to_owner:
                        layer, attr_name = var_to_owner[id(target_var)]
                    
                    if layer is None:
                        var_path = target_var.path if hasattr(target_var, 'path') else target_var.name
                        layer, attr_name = self._find_layer_by_path(shard_model, var_path)

                    if layer and attr_name:
                        self._replace_variable(layer, attr_name, target_var, sliced_val_tensor, device_id=device_id)
                    else:
                        # Fallback assign
                        target_var.assign(sliced_val_tensor)
                    
                    modified.add(name)
                    del source_val, sliced_val, sliced_val_tensor
                    gc.collect()

                    log_stats(f"After  {name}") # <--- LOG AFTER
        
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