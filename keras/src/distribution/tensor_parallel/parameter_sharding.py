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
        """Maps every variable ID to a LIST of (layer, attribute) pairs."""
        var_to_owners = {}
        stack = [model]
        visited = set()
        while stack:
            layer = stack.pop()
            if id(layer) in visited: continue
            visited.add(id(layer))

            # 1. Check tracking lists (CRITICAL for Functional models and weight tying)
            for lst_name in ['_trainable_weights', '_non_trainable_weights', '_weights']:
                if hasattr(layer, lst_name):
                    lst = getattr(layer, lst_name)
                    if isinstance(lst, list):
                        for v in lst:
                            if hasattr(v, 'assign') and hasattr(v, 'value'):
                                var_to_owners.setdefault(id(v), []).append((layer, lst_name))

            # 2. Check direct attributes
            for attr_name in dir(layer):
                if attr_name.startswith("__") or attr_name in ['trainable_variables', 'variables', 'weights']: 
                    continue
                try: attr_val = getattr(layer, attr_name, None)
                except Exception: continue
                
                if hasattr(attr_val, 'assign') and hasattr(attr_val, 'value'):
                    var_to_owners.setdefault(id(attr_val), []).append((layer, attr_name))
                elif hasattr(attr_val, 'layers') or hasattr(attr_val, 'weights'):
                    stack.append(attr_val)
                elif isinstance(attr_val, (list, tuple)):
                    for item in attr_val:
                        if hasattr(item, 'layers') or hasattr(item, 'weights'): 
                            stack.append(item)
        return var_to_owners

    def _replace_variable(self, layer, attr_name, old_var, new_var):
        """Swaps the variable and updates layer metadata to match sharded shapes."""
        # 1. Update List Elements
        if attr_name in ['_trainable_weights', '_non_trainable_weights', '_weights']:
            lst = getattr(layer, attr_name)
            for i, v in enumerate(lst):
                if v is old_var:
                    lst[i] = new_var
        else:
            # 2. Update Direct Attributes
            object.__setattr__(layer, attr_name, new_var)
            if not attr_name.startswith("_"):
                try: object.__setattr__(layer, "_" + attr_name, new_var)
                except: pass

        # 3. Update Layer Metadata (CRITICAL: prevents ops from expecting full shapes)
        if "Dense" in layer.__class__.__name__:
            # If we sharded the output dimension, update 'units'
            if new_var is getattr(layer, 'kernel', None) and new_var.shape[-1] != layer.units:
                layer.units = new_var.shape[-1]
        elif "Embedding" in layer.__class__.__name__:
            # If we sharded the embedding dimension, update 'output_dim'
            if new_var.shape[-1] != layer.output_dim:
                layer.output_dim = new_var.shape[-1]

    def shard_model_parameters(self, shard_model, weight_loader, config, device_id):
        modified_ids = set()
        old_to_new = {} # Cache to preserve weight tying
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

                    # 1. Use cached new_var if this is a tied weight
                    if id(target_var) in old_to_new:
                        new_var = old_to_new[id(target_var)]
                    else:
                        lookup_name = name.replace(f"shard_{self.rank}/", "")
                        raw_val = weight_loader(lookup_name)
                        if raw_val is None: continue
                        
                        # Apply sharding rule on CPU
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

                    # 2. Swap in all locations (Layer attributes and Model lists)
                    if id(target_var) in var_to_owners:
                        for owner, attr_name in var_to_owners[id(target_var)]:
                            self._replace_variable(owner, attr_name, target_var, new_var)

                    # 3. Shred the old CPU buffer
                    try: object.__setattr__(target_var, "_value", jax.numpy.zeros((0,), dtype=target_var.dtype))
                    except: pass
                    
                    modified_ids.add(id(target_var))
        
        # Collect names of variables that were sharded
        modified_names = {v.path if hasattr(v, 'path') else v.name for v in shard_model.variables if id(v) in modified_ids}
        return shard_model, modified_names

    def _find_matching_parameters(self, model, pattern: str):
        return [(v.path if hasattr(v, 'path') else v.name, v) 
                for v in model.variables 
                if re.search(pattern, v.path if hasattr(v, 'path') else v.name)]

def make_parameter_sharded_model(shard_model, weight_loader, config, rank, device_count, device_id):
    strategy = ParameterShardingStrategy(device_count, rank)
    return strategy.shard_model_parameters(shard_model, weight_loader, config, device_id)