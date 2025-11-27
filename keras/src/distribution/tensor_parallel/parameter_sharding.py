import logging
import gc
import re
import os
import psutil
from typing import Any, Tuple, Set
import numpy as np

from keras import Variable, device
from keras.src.models import Model
from keras.src.backend import distribution_lib
from keras.src.distribution.tensor_parallel.tensor_layout import split_tensor_for_parallelism

try:
    import jax
except ImportError:
    jax = None

logger = logging.getLogger(__name__)

def log_memory(stage=""):
    try:
        process = psutil.Process(os.getpid())
        mem = process.memory_info().rss / (1024 ** 3)
        print(f"   [MEM] {stage}: {mem:.2f} GB RAM used")
    except:
        pass

class ShardedWeight:
    def __init__(self, tensor_shard, name, trainable=True, device_id=None):
        current_dev = device_id if device_id else "cpu"
        
        # Sanitization
        if name:
            clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        else:
            clean_name = "sharded_var"
            
        with device(current_dev):
            # Explicitly pass dtype to prevent Keras from upcasting to float32
            self._variable = Variable(
                initializer=tensor_shard, 
                trainable=trainable, 
                name=clean_name,
                dtype=tensor_shard.dtype 
            )

    @property
    def name(self): return self._variable.name
    @property
    def value(self): return self._variable.value
    @property
    def numpy(self): return self._variable.numpy
    @property
    def shape(self): return self._variable.shape
    @property
    def dtype(self): return self._variable.dtype
    @property
    def trainable(self): return self._variable.trainable
    def assign(self, value): self._variable.assign(value)


class ParameterShardingStrategy:
    def __init__(self, device_count: int, rank: int):
        self.device_count = device_count
        self.rank = rank
        self.sharded_weights = {} 

    def shard_model_parameters(self, model, config, device_id: Any) -> Tuple["Model", Set[str]]:
        ParameterShardedModel = _define_parameter_sharded_model()
        
        print(f"🔧 Applying manual parameter sharding [v7-Trainable-Fix] (Rank {self.rank}/{self.device_count})...")
        log_memory("Start Sharding")
        
        modified_parameters = set()
        
        # Progress tracking
        total_items = 0
        for pattern, rule in config.items():
            matches = self._find_matching_parameters(model, pattern)
            total_items += len(matches)
        
        current_idx = 0

        for pattern, rule in config.items():
            matching_params = self._find_matching_parameters(model, pattern)

            for param_name, param in matching_params:
                current_idx += 1
                try:
                    split_dim = None
                    if isinstance(rule, (tuple, list)):
                        for axis_idx, axis_name in enumerate(rule):
                            if axis_name == "model":
                                split_dim = axis_idx
                                break
                    elif hasattr(rule, "axes"):
                        for axis_idx, axis_name in enumerate(rule.axes):
                            if axis_name == "model":
                                split_dim = axis_idx
                                break

                    if split_dim is not None:
                        print(f"   [{current_idx}/{total_items}] Sharding {param_name} | Trainable: {param.trainable}")
                        
                        # 1. Slice (CPU) - Returns bfloat16/float16
                        sharded_tensor_np = split_tensor_for_parallelism(
                            param, 
                            index=self.rank, 
                            device_count=self.device_count, 
                            dim=split_dim
                        )
                        
                        # 2. Upload to Device (Preserving Dtype AND Trainable Status)
                        sharded_var = ShardedWeight(
                            sharded_tensor_np, 
                            name=param_name, 
                            trainable=param.trainable, # <--- [CRITICAL FIX]
                            device_id=device_id
                        )
                        
                        self.sharded_weights[param_name] = sharded_var
                        modified_parameters.add(param_name)
                        
                        # 3. JAX Cleanup
                        if jax is not None:
                            try:
                                jax.block_until_ready(sharded_var.value)
                                jax.clear_caches()
                            except Exception:
                                pass

                        # 4. Host Cleanup
                        del sharded_tensor_np
                        try:
                            dummy = np.zeros((1,), dtype=param.dtype)
                            param.assign(dummy)
                        except:
                            pass
                        gc.collect()
                        
                        if current_idx % 20 == 0:
                            log_memory("After GC")
                        
                    else:
                        pass 

                except Exception as e:
                    print(f"   ❌ FATAL: Failed to shard {param_name}: {e}")
                    raise e

        sharded_model = ParameterShardedModel(
            original_model=model,
            sharding_strategy=self,
            config=config,
            device_id=device_id,
        )
        
        print(f"🎯 Parameter sharding completed.")
        return sharded_model, modified_parameters

    def _find_matching_parameters(self, model, pattern):
        matches = []
        for v in model.variables:
            if re.fullmatch(pattern, v.path) or re.fullmatch(pattern, v.name):
                matches.append((v.path, v))
        return matches


def _define_parameter_sharded_model():
    from keras.src.models import Model
    
    class ParameterShardedModel(Model):
        def __init__(self, original_model, sharding_strategy, config, device_id):
            super().__init__()
            self.original_model = original_model
            self.sharding_strategy = sharding_strategy
            self.config = config
            self.device_id = device_id
            self._recreate_variables()

        def _recreate_variables(self):
            self._sharded_vars = []
            for name, weight_obj in self.sharding_strategy.sharded_weights.items():
                self._sharded_vars.append(weight_obj)
                
        def call(self, inputs, training=None, mask=None):
            outputs = self.original_model(inputs, training=training, mask=mask)
            if hasattr(self.config, "output_rules"):
                for layer_name, rule in self.config.output_rules.items():
                    if rule == "allreduce sum":
                         outputs = distribution_lib.all_reduce(outputs, op="sum", axis_name="model")
            return outputs

    return ParameterShardedModel