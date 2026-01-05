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
        """Optimized mapping using internal Keras variable tracking."""
        var_to_owner = {}
        # Iterate through all layers and their specific variables to find the owner layer
        for layer in model._flatten_layers(include_self=True, recursive=True):
            # Check attributes to match variable names to layer attributes
            for attr_name in dir(layer):
                if attr_name.startswith("__"): continue
                try:
                    attr_val = getattr(layer, attr_name, None)
                    if hasattr(attr_val, 'assign') and hasattr(attr_val, 'value'):
                        var_to_owner[id(attr_val)] = (layer, attr_name)
                except Exception:
                    continue
        return var_to_owner

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

    def shard_model_parameters(self, shard_model, weight_loader, config, device_id):
        """Performs one-by-one sharding to minimize peak memory."""
        modified = set()
        var_to_owner = self._map_variables_to_owners(shard_model)
        
        # JAX specific optimization for device placement
        jax_target = None
        if keras.config.backend() == "jax":
            import jax
            try:
                d_str = str(device_id)
                idx = int(d_str.split(":")[-1]) if ":" in d_str else 0
                jax_target = jax.devices()[idx]
            except Exception: pass

        # Iterate through rules and shard matching variables
        for pattern, action in config.state_rules.items():
            if not callable(action): continue
            
            # Find matching variables in the model
            for target_var in shard_model.variables:
                name = target_var.path if hasattr(target_var, 'path') else target_var.name
                if re.search(pattern, name):
                    # Load the large parameter from disk/memory-mapped file
                    source_val = weight_loader(name)
                    if source_val is None: continue

                    # Generate the shard
                    sliced_val = action(source_val, self.rank)
                    
                    # Move to GPU/TPU device
                    if jax_target is not None:
                        import jax
                        sliced_val_tensor = jax.device_put(sliced_val, jax_target).astype(target_var.dtype)
                    else:
                        with keras.device(device_id):
                            sliced_val_tensor = ops.convert_to_tensor(sliced_val, dtype=target_var.dtype)

                    # Swap the variable in the model
                    if id(target_var) in var_to_owner:
                        layer, attr_name = var_to_owner[id(target_var)]
                        self._replace_variable(layer, attr_name, target_var, sliced_val_tensor, device_id)
                    else:
                        target_var.assign(sliced_val_tensor)
                    
                    modified.add(name)
                    
                    # Immediate cleanup: delete references and trigger GC to free RAM
                    del source_val, sliced_val, sliced_val_tensor
                    gc.collect()
        
        return shard_model, modified

def make_parameter_sharded_model(shard_model, weight_loader, config, rank, device_count, device_id):
    strategy = ParameterShardingStrategy(device_count, rank)
    return strategy.shard_model_parameters(shard_model, weight_loader, config, device_id)