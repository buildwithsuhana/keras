# buildwithsuhana/keras/keras-ananta/keras/src/distribution/tensor_parallel/parameter_sharding.py

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

# Registers bfloat16 with numpy to avoid the |V2 error
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
        var_to_owners = {}
        stack = [model]
        visited = set()
        while stack:
            layer = stack.pop()
            if id(layer) in visited: continue
            visited.add(id(layer))
            for attr_name in dir(layer):
                if attr_name.startswith("__"): continue
                try: attr_val = getattr(layer, attr_name, None)
                except: continue
                is_var = hasattr(attr_val, 'assign') and hasattr(attr_val, 'value')
                if is_var:
                    var_to_owners.setdefault(id(attr_val), []).append((layer, attr_name))
                elif hasattr(attr_val, 'weights') and not is_var:
                    stack.append(attr_val)
                elif isinstance(attr_val, (list, tuple)):
                    for item in attr_val:
                        if hasattr(item, 'weights'): stack.append(item)
            if hasattr(layer, 'layers'):
                try: stack.extend(layer.layers)
                except: pass
        return var_to_owners

    def _replace_variable(self, layer, attr_name, old_var, new_val_tensor, device_id=None):
        """Creates GPU variable and swap-destroys the CPU one to break Host RAM references."""
        try:
            with keras.device(device_id):
                new_var = keras.Variable(
                    initializer=new_val_tensor,
                    shape=new_val_tensor.shape,
                    dtype=old_var.dtype,
                    trainable=old_var.trainable,
                    name=old_var.name 
                )
        except: return old_var
        
        object.__setattr__(layer, attr_name, new_var)
        if not attr_name.startswith("_"):
            try: object.__setattr__(layer, "_" + attr_name, new_var)
            except: pass

        for lst_name in ['_trainable_weights', '_non_trainable_weights', '_weights']:
            if hasattr(layer, lst_name):
                lst = getattr(layer, lst_name)
                if isinstance(lst, list):
                    for i, v in enumerate(lst):
                        if v is old_var: lst[i] = new_var 

        # DESTRUCTIVE: Shrink CPU buffer to release 18GB Host RAM allocation
        try:
            old_var.assign(np.zeros((1,), dtype=old_var.dtype))
        except: pass
        return new_var

    def shard_model_parameters(self, shard_model, weight_loader, config, device_id):
        modified = set()
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
                    lookup_name = name.replace(f"shard_{self.rank}/", "")
                    try:
                        source_val = weight_loader(lookup_name)
                        if source_val is None: continue
                        if hasattr(source_val, 'dtype') and (str(source_val.dtype).startswith('|V') or source_val.dtype == np.dtype('V2')):
                            source_val = source_val.view(ml_dtypes.bfloat16) if ml_dtypes else source_val.astype('float32')
                    except: continue

                    sliced_val = action(source_val, self.rank)
                    sliced_val_tensor = jax.device_put(sliced_val, jax_target)
                    sliced_val_tensor = sliced_val_tensor.astype(target_var.dtype)

                    if id(target_var) in var_to_owners:
                        for layer, attr_name in var_to_owners[id(target_var)]:
                            self._replace_variable(layer, attr_name, target_var, sliced_val_tensor, device_id=device_id)
                    else:
                        target_var.assign(sliced_val_tensor)
                    
                    modified.add(name)
                    del source_val, sliced_val, sliced_val_tensor
                    gc.collect()
                    # NEW: Per-variable memory log
                    log_stats(f"Sharded {name}")
        
        return shard_model, modified

    def _find_matching_parameters(self, model, pattern: str):
        return [(v.path if hasattr(v, 'path') else v.name, v) for v in model.variables if re.search(pattern, v.path if hasattr(v, 'path') else v.name)]

def make_parameter_sharded_model(shard_model, weight_loader, config, rank, device_count, device_id):
    strategy = ParameterShardingStrategy(device_count, rank)
    return strategy.shard_model_parameters(shard_model, weight_loader, config, device_id)