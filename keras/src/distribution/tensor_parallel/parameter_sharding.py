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
        # Recursive scan of all layers
        for layer in model._flatten_layers(include_self=True, recursive=True):
            # Hub layers often store weights in private lists or dicts
            search_targets = [layer]
            if hasattr(layer, '_layers'): search_targets.extend(layer._layers)
            
            for target in search_targets:
                for attr_name in dir(target):
                    if attr_name.startswith("__"): continue
                    try:
                        val = getattr(target, attr_name, None)
                        if hasattr(val, 'assign') and hasattr(val, 'value'):
                            var_to_owner[id(val)] = (target, attr_name)
                    except: continue
        return var_to_owner

    def shard_model_parameters(self, shard_model, weight_loader, config, device_id):
        modified = set()
        var_to_owner = self._map_variables_to_owners(shard_model)
        
        # Determine rules
        for target_var in shard_model.variables:
            name = target_var.path
            action = None
            for pattern, act in config.state_rules.items():
                if re.search(pattern, name):
                    action = act
                    break
            
            if not action or not callable(action):
                continue

            # Load and slice
            source_val = weight_loader(name)
            if source_val is None: continue
            sliced_val = action(source_val, self.rank)

            with keras.device(device_id):
                # Ensure we have a tensor on the correct device
                sliced_val_tensor = ops.convert_to_tensor(sliced_val, dtype=target_var.dtype)

            # FIND PARENT AND REPLACE
            if id(target_var) in var_to_owner:
                layer, attr_name = var_to_owner[id(target_var)]
                self._replace_variable(layer, attr_name, target_var, sliced_val_tensor, device_id)
                modified.add(name)
            else:
                # CRITICAL: If no owner found, we must still update the shape to prevent 
                # broadcasting errors like 'Incompatible shapes: (1792,) and (3584,)'
                print(f"🚨 Forced replacement for ownerless var: {name}")
                with keras.device(device_id):
                    new_var = keras.Variable(sliced_val_tensor, dtype=target_var.dtype, name=target_var.name + "_forced")
                    # Find and replace in model's internal tracking lists
                    for lst in [shard_model._trainable_weights, shard_model._non_trainable_weights]:
                        for i, v in enumerate(lst):
                            if v is target_var: lst[i] = new_var
                modified.add(name)
            
            del source_val, sliced_val
            gc.collect()
        
        return shard_model, modified

    def _replace_variable(self, layer, attr_name, old_var, new_val_tensor, device_id=None):
        """Swaps a property-based variable with its sharded version by bypassing setters."""
        print(f"      🛠️  [Swapping] '{attr_name}' on '{layer.name}' -> {device_id}")
        
        new_name = f"{old_var.name}_shard_{self.rank}"
        
        with keras.device(device_id):
            new_var = keras.Variable(
                initializer=new_val_tensor,
                shape=new_val_tensor.shape,
                dtype=old_var.dtype,
                trainable=old_var.trainable,
                name=new_name 
            )
        
        # FIX: Bypass properties by writing directly to the instance __dict__
        # This handles 'EinsumDense' kernel/bias property issues.
        layer.__dict__[attr_name] = new_var

        # If it was a private attribute (e.g. _kernel), ensure that is set too
        if not attr_name.startswith("_"):
            layer.__dict__["_" + attr_name] = new_var
        
        # Update internal Keras tracking lists
        for lst_name in ['_trainable_weights', '_non_trainable_weights', '_weights']:
            if hasattr(layer, lst_name):
                lst = getattr(layer, lst_name)
                if isinstance(lst, list):
                    for i, v in enumerate(lst):
                        if v is old_var:
                            lst[i] = new_var
        return new_var

def make_parameter_sharded_model(shard_model, weight_loader, config, rank, device_count, device_id):
    strategy = ParameterShardingStrategy(device_count, rank)
    return strategy.shard_model_parameters(shard_model, weight_loader, config, device_id)