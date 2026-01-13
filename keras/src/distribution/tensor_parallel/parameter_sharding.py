import logging
import re
import gc
import os
import psutil
from typing import Any, Tuple, Set, Callable
import keras
import numpy as np
import jax

class ParameterShardingStrategy:
    def __init__(self, device_count: int, rank: int):
        self.device_count = device_count
        self.rank = rank

    def _map_variables_to_owners(self, model):
        """Exhaustively maps variable IDs to all layers and internal lists that reference them."""
        var_to_owners = {}
        stack, visited = [model], set()
        WEIGHT_LISTS = ['_trainable_weights', '_non_trainable_weights', '_weights', '_variables']
        while stack:
            layer = stack.pop()
            if id(layer) in visited: continue
            visited.add(id(layer))
            for attr_name, attr_val in layer.__dict__.items():
                if attr_name.startswith("__"): continue
                if hasattr(attr_val, 'assign') and hasattr(attr_val, 'value'):
                    var_to_owners.setdefault(id(attr_val), []).append((layer, attr_name, None))
                if hasattr(attr_val, 'layers') or hasattr(attr_val, 'weights') or hasattr(attr_val, '_layers'):
                    stack.append(attr_val)
                elif isinstance(attr_val, (list, tuple)):
                    for item in attr_val:
                        if hasattr(item, 'layers') or hasattr(item, 'weights') or hasattr(item, '_layers'):
                            stack.append(item)
            for lst_name in WEIGHT_LISTS:
                if hasattr(layer, lst_name):
                    lst = getattr(layer, lst_name)
                    if isinstance(lst, list):
                        for i, v in enumerate(lst):
                            if hasattr(v, 'assign') and hasattr(v, 'value'):
                                var_to_owners.setdefault(id(v), []).append((layer, lst_name, i))
        return var_to_owners

    def _replace_variable(self, layer, attr_name, old_var, new_var, index=None):
        """Swaps variable objects and updates layer metadata for sharded shapes."""
        if index is not None:
            lst = getattr(layer, attr_name)
            if isinstance(lst, list) and index < len(lst) and lst[index] is old_var:
                lst[index] = new_var
            return
        try:
            object.__setattr__(layer, attr_name, new_var)
            if not attr_name.startswith("_"):
                try: object.__setattr__(layer, "_" + attr_name, new_var)
                except: pass
        except: pass
        # Update layer metadata to prevent 'Build' or 'Shape' errors
        if hasattr(layer, "output_dim") and "Embedding" in layer.__class__.__name__:
            layer.output_dim = new_var.shape[-1]
        elif hasattr(layer, "units") and ("Dense" in layer.__class__.__name__):
            layer.units = new_var.shape[-1]

    def shard_model_parameters(self, shard_model, weight_loader, config, device_id):
        modified_paths = set() 
        var_to_owners = self._map_variables_to_owners(shard_model)
        d_idx = int(str(device_id).split(":")[-1]) if ":" in str(device_id) else 0
        jax_target = jax.devices('gpu')[d_idx]

        for pattern, action in config.state_rules.items():
            targets = self._find_matching_parameters(shard_model, pattern)
            for name, target_var in targets:
                # FIX: Check path instead of ID to avoid re-sharding or skipping replaced vars
                if name in modified_paths: continue

                lookup_name = re.sub(r'^shard_model_\d+/', '', name)
                raw_val = weight_loader(lookup_name)
                if raw_val is None: continue
                
                if hasattr(raw_val, 'dtype') and ("V" in str(raw_val.dtype) or "void" in str(raw_val.dtype)):
                    import ml_dtypes
                    raw_val = raw_val.view(ml_dtypes.bfloat16)

                sliced_val = action(raw_val, self.rank)
                val_gpu = jax.device_put(sliced_val, jax_target)
                
                with keras.device(device_id):
                    new_var = keras.Variable(val_gpu, dtype=target_var.dtype, name=target_var.name)

                if id(target_var) in var_to_owners:
                    for owner, attr_name, index in var_to_owners[id(target_var)]:
                        self._replace_variable(owner, attr_name, target_var, new_var, index=index)

                # Memory Logging
                mem_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
                print(f"⛓️ [Rank {self.rank}] Sharded variable: {name} | Host RSS: {mem_mb:.0f} MB")

                try: object.__setattr__(target_var, "_value", jax.numpy.zeros((0,), dtype=target_var.dtype))
                except: pass
                modified_paths.add(name)
        
        return shard_model, modified_paths
    
    def _find_matching_parameters(self, model, pattern: str):
        return [(v.path if hasattr(v, 'path') else v.name, v) for v in model.variables 
                if re.search(pattern, v.path if hasattr(v, 'path') else v.name)]

def make_parameter_sharded_model(shard_model, weight_loader, config, rank, device_count, device_id):
    strategy = ParameterShardingStrategy(device_count, rank)
    return strategy.shard_model_parameters(shard_model, weight_loader, config, device_id)