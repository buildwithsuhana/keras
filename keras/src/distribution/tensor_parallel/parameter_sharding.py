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

# CRITICAL: Registers bfloat16 with numpy to prevent the |V2 (void) dtype error
try:
    import ml_dtypes
except ImportError:
    ml_dtypes = None

if TYPE_CHECKING:
    from keras.src.models import Model

logger = logging.getLogger(__name__)

def log_stats(stage=""):
    """Logs current system RAM and GPU VRAM usage."""
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 ** 2)
    gpu_str = ""
    try:
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'], encoding='utf-8')
        mems = [int(x) for x in result.strip().split('\n') if x.strip()]
        for i, m in enumerate(mems): gpu_str += f"G{i}:{m}M "
    except: pass
    print(f"   📊 [Stats] {stage} | RAM: {mem_mb:.0f}MB | {gpu_str}")

class ParameterShardingStrategy:
    def __init__(self, device_count: int, rank: int):
        self.device_count = device_count
        self.rank = rank

    def _map_variables_to_owners(self, model):
        """Maps every variable ID to a LIST of (owner, attribute_name, index_if_list)."""
        var_to_owners = {}
        stack = [model]
        visited = set()
        
        # Internal lists that should be searched but never overwritten via setattr
        WEIGHT_LISTS = ['_trainable_weights', '_non_trainable_weights', '_weights']

        while stack:
            layer = stack.pop()
            if id(layer) in visited: continue
            visited.add(id(layer))

            # 1. Search attributes in __dict__ (avoids properties)
            for attr_name, attr_val in layer.__dict__.items():
                if attr_name.startswith("__"): continue
                
                if hasattr(attr_val, 'assign') and hasattr(attr_val, 'value'):
                    var_to_owners.setdefault(id(attr_val), []).append((layer, attr_name, None))
                
                # Recurse into sub-layers or lists of layers
                if hasattr(attr_val, 'layers') or hasattr(attr_val, 'weights'):
                    stack.append(attr_val)
                elif isinstance(attr_val, (list, tuple)):
                    for item in attr_val:
                        if hasattr(item, 'layers') or hasattr(item, 'weights'):
                            stack.append(item)

            # 2. Search Keras internal weight lists
            for lst_name in WEIGHT_LISTS:
                if hasattr(layer, lst_name):
                    lst = getattr(layer, lst_name)
                    if isinstance(lst, list):
                        for i, v in enumerate(lst):
                            if hasattr(v, 'assign') and hasattr(v, 'value'):
                                var_to_owners.setdefault(id(v), []).append((layer, lst_name, i))
        return var_to_owners

    def _replace_variable(self, layer, attr_name, old_var, new_var, index=None):
        """Swaps variable objects and updates layer metadata for shape consistency."""
        # 1. Update List Elements (prevents overwriting the list itself)
        if index is not None:
            lst = getattr(layer, attr_name)
            if isinstance(lst, list) and index < len(lst) and lst[index] is old_var:
                lst[index] = new_var
            return

        # 2. Update Direct Attributes
        try:
            object.__setattr__(layer, attr_name, new_var)
            if not attr_name.startswith("_"):
                try: object.__setattr__(layer, "_" + attr_name, new_var)
                except: pass
        except: pass

        # 3. Update Metadata (Prevents shape mismatch errors in layer logic)
        if hasattr(layer, "output_dim") and "Embedding" in layer.__class__.__name__:
            if new_var.shape[-1] != layer.output_dim:
                layer.output_dim = new_var.shape[-1]
        elif hasattr(layer, "units") and new_var is getattr(layer, 'kernel', None):
            if new_var.shape[-1] != layer.units:
                layer.units = new_var.shape[-1]

    def shard_model_parameters(self, shard_model, weight_loader, config, device_id):
        modified_ids = set()
        old_to_new = {} 
        var_to_owners = self._map_variables_to_owners(shard_model)
        
        jax_target = None
        import jax
        try:
            d_idx = int(str(device_id).split(":")[-1]) if ":" in str(device_id) else 0
            jax_target = jax.devices('gpu')[d_idx]
        except: pass

        for pattern, action in config.state_rules.items():
            if callable(action):
                targets = self._find_matching_parameters(shard_model, pattern)
                for name, target_var in targets:
                    if id(target_var) in modified_ids: continue

                    if id(target_var) in old_to_new:
                        new_var = old_to_new[id(target_var)]
                    else:
                        lookup_name = name.replace(f"shard_{self.rank}/", "")
                        raw_val = weight_loader(lookup_name)
                        if raw_val is None: continue
                        
                        # Restore bfloat16 if needed
                        if hasattr(raw_val, 'dtype') and ("V" in str(raw_val.dtype) or "void" in str(raw_val.dtype)):
                            import ml_dtypes
                            raw_val = raw_val.view(ml_dtypes.bfloat16)

                        sliced_val = action(raw_val, self.rank)
                        val_gpu = jax.device_put(sliced_val, jax_target)
                        
                        with keras.device(device_id):
                            new_var = keras.Variable(
                                initializer=val_gpu,
                                shape=val_gpu.shape,
                                dtype=target_var.dtype,
                                trainable=target_var.trainable,
                                name=target_var.name
                            )
                        old_to_new[id(target_var)] = new_var

                    # Replace variable in all owner locations
                    if id(target_var) in var_to_owners:
                        for owner, attr_name, index in var_to_owners[id(target_var)]:
                            self._replace_variable(owner, attr_name, target_var, new_var, index=index)

                    # Shred CPU RAM
                    try: object.__setattr__(target_var, "_value", jax.numpy.zeros((0,), dtype=target_var.dtype))
                    except: pass
                    
                    modified_ids.add(id(target_var))
                    log_stats(f"Sharded {name}")
        
        all_sharded_paths = {v.path if hasattr(v, 'path') else v.name 
                            for v in shard_model.variables if id(v) in modified_ids}
        return shard_model, all_sharded_paths
    
    def _find_matching_parameters(self, model, pattern: str):
        return [(v.path if hasattr(v, 'path') else v.name, v) 
                for v in model.variables 
                if re.search(pattern, v.path if hasattr(v, 'path') else v.name)]

def make_parameter_sharded_model(shard_model, weight_loader, config, rank, device_count, device_id):
    strategy = ParameterShardingStrategy(device_count, rank)
    return strategy.shard_model_parameters(shard_model, weight_loader, config, device_id)