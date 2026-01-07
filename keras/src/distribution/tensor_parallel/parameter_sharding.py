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

def log_stats(stage=""):
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
        """Robust mapping specifically for Keras Hub / Gemma models."""
        var_to_owner = {}
        # Recursive scan of all layers to find which layer owns which Variable
        for layer in model._flatten_layers(include_self=True, recursive=True):
            # Scan __dict__ directly to find variables (bypasses property logic)
            for attr_name, val in layer.__dict__.items():
                if id(val) in var_to_owner: continue
                if hasattr(val, 'assign') and hasattr(val, 'value'):
                    var_to_owner[id(val)] = (layer, attr_name)
            
            # Special check for internal tracking lists (handles hidden Hub weights)
            for lst_name in ['_trainable_weights', '_non_trainable_weights', '_weights', 'weights']:
                if hasattr(layer, lst_name):
                    weights_list = getattr(layer, lst_name)
                    if not isinstance(weights_list, list): continue
                    for weight in weights_list:
                        if id(weight) not in var_to_owner:
                            # Extract clean attribute name from path (e.g. 'embeddings' from 'token_embedding/embeddings')
                            attr_name = weight.path.split('/')[-1] if hasattr(weight, 'path') else weight.name.split(':')[0]
                            var_to_owner[id(weight)] = (layer, attr_name)
        return var_to_owner

    # Inside ParameterShardingStrategy in parameter_sharding.py

    def shard_model_parameters(self, shard_model, weight_loader, config, device_id):
        modified = set()
        var_to_owner = self._map_variables_to_owners(shard_model)
        
        for target_var in shard_model.variables:
            name = target_var.path
            action = next((act for pat, act in config.state_rules.items() if re.search(pat, name)), None)
            
            if not action or not callable(action):
                continue

            source_val = weight_loader(name)
            if source_val is None: continue
            sliced_val = action(source_val, self.rank)

            with keras.device(device_id):
                sliced_val_tensor = ops.convert_to_tensor(sliced_val, dtype=target_var.dtype)

            # --- CRITICAL FIX: NAVIGATE TO THE ACTUAL LEAF LAYER ---
            owner_found = False
            parts = name.split('/')
            curr_obj = shard_model
            
            try:
                # Walk the path: decoder_block_0 -> attention -> query
                for part in parts[:-1]:
                    if hasattr(curr_obj, part):
                        curr_obj = getattr(curr_obj, part)
                    else:
                        # Fallback for hidden attributes (e.g. _token_embedding)
                        curr_obj = getattr(curr_obj, f"_{part}")
                
                attr_name = parts[-1]
                # Replace the variable on the specific leaf object
                self._replace_variable(curr_obj, attr_name, target_var, sliced_val_tensor, device_id)
                owner_found = True
                print(f"      ✅ [Mapped] '{attr_name}' on layer '{curr_obj.name}' to {device_id}")
            except Exception:
                # Fallback to the ID-based owner map if path navigation fails
                if id(target_var) in var_to_owner:
                    layer, attr_name = var_to_owner[id(target_var)]
                    self._replace_variable(layer, attr_name, target_var, sliced_val_tensor, device_id)
                    owner_found = True

            if not owner_found:
                # Last resort: direct assignment
                with keras.device(device_id):
                    target_var.assign(sliced_val_tensor)
            
            modified.add(name)
            del source_val, sliced_val
            gc.collect()
        return shard_model, modified

    def _replace_variable(self, layer, attr_name, old_var, new_val_tensor, device_id=None):
        """Swaps a variable with its sharded version and forces GPU placement."""
        # Sanitize attribute name
        attr_name = attr_name.replace(":", "_").split("/")[-1]
        
        with keras.device(device_id):
            new_var = keras.Variable(
                initializer=new_val_tensor,
                shape=new_val_tensor.shape,
                dtype=old_var.dtype,
                trainable=old_var.trainable,
                name=f"{old_var.name}_shard_{self.rank}"
            )
        
        # 1. Update the instance attribute (Bypass property setters)
        layer.__dict__[attr_name] = new_var
        if not attr_name.startswith("_"):
            layer.__dict__["_" + attr_name] = new_var
        
        # 2. Update Keras internal tracking lists to ensure shard.trainable_variables is updated
        for lst_name in ['_trainable_weights', '_non_trainable_weights', '_weights']:
            if hasattr(layer, lst_name):
                lst = getattr(layer, lst_name)
                if isinstance(lst, list):
                    for i, v in enumerate(lst):
                        if v is old_var:
                            lst[i] = new_var
                            
        print(f"      ✅ [Mapped] '{attr_name}' on layer '{layer.name}' to {device_id}")
        return new_var

def make_parameter_sharded_model(shard_model, weight_loader, config, rank, device_count, device_id):
    strategy = ParameterShardingStrategy(device_count, rank)
    return strategy.shard_model_parameters(shard_model, weight_loader, config, device_id)