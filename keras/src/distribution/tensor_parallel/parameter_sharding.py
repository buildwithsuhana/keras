import logging
import re
import gc
import os
import psutil
import subprocess
import keras
from keras import ops
import numpy as np

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
                except: continue
                if attr_val is None: continue
                is_var = hasattr(attr_val, 'assign') and hasattr(attr_val, 'value')
                if is_var: layer_vars.setdefault(id(attr_val), []).append(attr_name)
                elif hasattr(attr_val, 'weights') and not is_var: stack.append(attr_val)
                elif isinstance(attr_val, (list, tuple)):
                    for item in attr_val:
                        if hasattr(item, 'weights'): stack.append(item)
            for vid, names in layer_vars.items():
                best_name = names[0]
                for name in names:
                    if name.startswith("_"): best_name = name; break
                var_to_owner[vid] = (layer, best_name)
            if hasattr(layer, 'layers'): stack.extend(layer.layers)
        return var_to_owner

    def _replace_variable(self, layer, attr_name, old_var, new_val_tensor, device_id=None):
        """
        FIX: Replace the logical variable and force Keras tracking lists to refresh on GPU.
        """
        new_name = f"{old_var.name.split(':')[0]}_shard_{self.rank}"
        
        with keras.device(device_id):
            new_var = keras.Variable(
                initializer=new_val_tensor,
                shape=new_val_tensor.shape,
                dtype=old_var.dtype,
                trainable=old_var.trainable,
                name=new_name 
            )

        # Primary attribute replacement
        object.__setattr__(layer, attr_name, new_var)

        # FIXED: Synchronize Keras' internal weight tracking lists
        for lst_name in ['_trainable_weights', '_non_trainable_weights', '_weights']:
            if hasattr(layer, lst_name):
                lst = getattr(layer, lst_name)
                if isinstance(lst, list):
                    for i, v in enumerate(lst):
                        if v is old_var: lst[i] = new_var
        return new_var

    def shard_model_parameters(self, shard_model, weight_loader, config, device_id):
        modified = set()
        var_to_owner = self._map_variables_to_owners(shard_model)
        
        import jax
        d_str = str(device_id)
        idx = int(d_str.split(":")[-1]) if ":" in d_str else 0
        try: jax_target = jax.devices('gpu')[idx]
        except: jax_target = jax.devices()[idx]

        for pattern, action in config.state_rules.items():
            if callable(action):
                targets = self._find_matching_parameters(shard_model, pattern)
                for name, target_var in targets:
                    source_val = weight_loader(name)
                    if source_val is None: continue

                    sliced_val = action(source_val, self.rank)
                    
                    # FIXED: Transfer tensor to GPU before wrapping in Keras Variable
                    sliced_val_tensor = jax.device_put(sliced_val, jax_target).astype(target_var.dtype)

                    layer, attr_name = var_to_owner.get(id(target_var), (None, None))
                    if layer and attr_name:
                        self._replace_variable(layer, attr_name, target_var, sliced_val_tensor, device_id=device_id)
                        modified.add(name)
                    
                    # Explicit JAX sync to avoid memory fragmentation
                    sliced_val_tensor.block_until_ready()
                    gc.collect()
        
        return shard_model, modified

    def _find_matching_parameters(self, model, pattern: str):
        matches = []
        for v in model.variables:
            name = v.path if hasattr(v, 'path') else v.name
            if re.search(pattern, name): matches.append((name, v))
        return matches

def make_parameter_sharded_model(shard_model, weight_loader, config, rank, device_count, device_id):
    strategy = ParameterShardingStrategy(device_count, rank)
    return strategy.shard_model_parameters(shard_model, weight_loader, config, device_id)