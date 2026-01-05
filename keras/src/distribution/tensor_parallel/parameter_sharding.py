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
            # Scan attributes for variables
            for attr_name in dir(layer):
                if attr_name.startswith("__"): continue
                try:
                    val = getattr(layer, attr_name, None)
                    if hasattr(val, 'assign') and hasattr(val, 'value'):
                        var_to_owner[id(val)] = (layer, attr_name)
                except: continue
            
            # Special check for internal tracking lists (handles hidden Hub weights)
            if hasattr(layer, 'weights'):
                for weight in layer.weights:
                    if id(weight) not in var_to_owner:
                        # Extract clean attribute name (e.g. 'embeddings:0' -> 'embeddings')
                        attr_name = weight.name.split('/')[-1].split(':')[0]
                        var_to_owner[id(weight)] = (layer, attr_name)
        return var_to_owner

    def shard_model_parameters(self, shard_model, weight_loader, config, device_id):
        """Performs one-by-one sharding to minimize peak memory."""
        modified = set()
        var_to_owner = self._map_variables_to_owners(shard_model)
        
        # Determine sharding rules for each variable in the model
        for target_var in shard_model.variables:
            name = target_var.path if hasattr(target_var, 'path') else target_var.name
            action = None
            for pattern, act in config.state_rules.items():
                if re.search(pattern, name):
                    action = act
                    break
            
            if not action or not callable(action):
                continue

            # Load the full parameter from disk and slice it on CPU
            source_val = weight_loader(name)
            if source_val is None: continue
            sliced_val = action(source_val, self.rank)

            with keras.device(device_id):
                # Ensure the sharded tensor is moved to the target device memory
                sliced_val_tensor = ops.convert_to_tensor(sliced_val, dtype=target_var.dtype)

            # FIND PARENT AND REPLACE (Handles shape changes)
            if id(target_var) in var_to_owner:
                layer, attr_name = var_to_owner[id(target_var)]
                self._replace_variable(layer, attr_name, target_var, sliced_val_tensor, device_id)
                modified.add(name)
            else:
                # Handle ownerless variables (often top-level model weights)
                print(f"🚨 Forced replacement for ownerless var: {name}")
                with keras.device(device_id):
                    new_var = keras.Variable(
                        sliced_val_tensor, 
                        dtype=target_var.dtype, 
                        trainable=target_var.trainable,
                        name=target_var.name + "_shard"
                    )
                
                # Update model-level tracking properties safely
                for attr in ["trainable_variables", "variables", "non_trainable_variables"]:
                    if hasattr(shard_model, attr):
                        try:
                            lst = getattr(shard_model, attr)
                            for i, v in enumerate(lst):
                                if v.path == target_var.path:
                                    lst[i] = new_var
                        except: continue
                modified.add(name)
            
            # Aggressive cleanup to avoid OOM during the sharding loop
            del source_val, sliced_val
            gc.collect()
        
        return shard_model, modified

    def _replace_variable(self, layer, attr_name, old_var, new_val_tensor, device_id=None):
        """Swaps a variable with its sharded version by bypassing read-only properties."""
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
        
        # Bypass potential property setters (common in Hub layers) by writing to __dict__
        layer.__dict__[attr_name] = new_var
        if not attr_name.startswith("_"):
            layer.__dict__["_" + attr_name] = new_var
        
        # Update layer's internal weight tracking lists
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