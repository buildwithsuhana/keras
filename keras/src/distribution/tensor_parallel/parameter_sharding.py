import logging
import gc
import re
import os
import psutil
from typing import Any, Tuple, Set
import numpy as np

from keras import Variable, device, layers
from keras.src.models import Model
from keras.src.backend import distribution_lib
from keras.src.distribution.tensor_parallel.tensor_layout import split_tensor_for_parallelism

try:
    import jax
    from jax.sharding import Mesh, PartitionSpec, NamedSharding
    from jax.experimental import mesh_utils
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
    """Wrapper that behaves like a Keras Variable but holds a distributed JAX array."""
    def __init__(self, tensor_val, name, trainable=True):
        if name:
            clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        else:
            clean_name = "sharded_var"
            
        # SPMD: tensor_val is already a global jax.Array
        self._variable = Variable(
            initializer=tensor_val, 
            trainable=trainable, 
            name=clean_name,
            dtype=tensor_val.dtype
        )

    # Proxy all attribute access to the internal variable
    def __getattr__(self, name):
        return getattr(self._variable, name)

class ParameterShardingStrategy:
    def __init__(self, device_count: int, rank: int):
        self.device_count = device_count
        self.rank = rank
        self.sharded_weights = {} 
        
        # SPMD Check
        self.local_devices = jax.local_devices() if jax else []
        self.is_spmd = len(self.local_devices) >= self.device_count
        
        if self.is_spmd:
            print(f"   [Strategy] SPMD Mode: Managing {len(self.local_devices)} local GPUs.")
            self.mesh = Mesh(mesh_utils.create_device_mesh((1, self.device_count)), axis_names=('batch', 'model'))
        else:
            print(f"   [Strategy] MPMD Mode: Managing Rank {self.rank}.")

    def shard_model_parameters(self, model, config, device_id: Any) -> Tuple["Model", Set[str]]:
        ParameterShardedModel = _define_parameter_sharded_model()
        
        print(f"🔧 Applying parameter sharding [v10-InPlace-Swap]...")
        log_memory("Start")
        
        modified_parameters = set()
        
        # We traverse layers recursively to find and replace variables IN-PLACE.
        # This ensures the model actually uses the shards and drops the old heavy variables.
        self._recursive_shard_layers(model, config, modified_parameters)

        sharded_model = ParameterShardedModel(
            original_model=model,
            sharding_strategy=self,
            config=config,
            device_id=device_id,
        )
        
        print(f"🎯 Sharding Complete. Replaced {len(modified_parameters)} parameters.")
        log_memory("Final")
        return sharded_model, modified_parameters

    def _recursive_shard_layers(self, current_layer, config, modified_set):
        """Recursively walks layers and swaps weights matching config rules."""
        
        # 1. Process Children First (Depth-First)
        if hasattr(current_layer, 'layers'):
            for sub_layer in current_layer.layers:
                self._recursive_shard_layers(sub_layer, config, modified_set)
                
        # 2. Process Current Layer's Weights
        # We iterate over attributes to find the name (e.g. 'kernel') corresponding to the weight
        # This allows us to do setattr(layer, 'kernel', new_var)
        
        # Helper: Get all variables in this layer
        layer_vars = []
        if hasattr(current_layer, "weights"):
            layer_vars = current_layer.weights
        
        if not layer_vars:
            return

        for variable in layer_vars:
            # Check if this variable has a rule in the LayoutMap
            # We match by path or name
            rule = config.get(variable.path, config.get(variable.name, None))
            
            if rule is None:
                # Try regex matching from config keys
                for pattern, r in config.items():
                    if isinstance(pattern, str) and (re.fullmatch(pattern, variable.path) or re.fullmatch(pattern, variable.name)):
                        rule = r
                        break
            
            if rule is not None:
                # We found a rule! Time to shard and swap.
                self._shard_and_swap(current_layer, variable, rule, modified_set)

    def _shard_and_swap(self, layer, variable, rule, modified_set):
        param_name = variable.name
        
        if param_name in modified_set:
            return # Already processed (shared weight)

        try:
            # 1. Interpret Rule
            split_dim = None
            if isinstance(rule, (tuple, list)):
                for i, axis in enumerate(rule):
                    if axis == "model": split_dim = i; break
            elif hasattr(rule, "axes"):
                for i, axis in enumerate(rule.axes):
                    if axis == "model": split_dim = i; break

            if split_dim is None:
                return

            print(f"   [Swap] Processing {param_name}...")

            # 2. Force Freeze Logic
            is_trainable = False
            if "lora" in param_name.lower():
                is_trainable = True
                print(f"      🔥 Keeping LoRA Trainable")

            # 3. Create Sharded Array (SPMD)
            if self.is_spmd:
                device_buffers = []
                for dev_idx in range(self.device_count):
                    # Slice on CPU
                    shard_np = split_tensor_for_parallelism(
                        variable, index=dev_idx, device_count=self.device_count, dim=split_dim
                    )
                    # Push to GPU
                    dev_buffer = jax.device_put(shard_np, self.local_devices[dev_idx])
                    device_buffers.append(dev_buffer)
                    del shard_np
                
                # Combine
                spec = PartitionSpec(None, 'model') if split_dim == 1 else PartitionSpec('model', None)
                out_sharding = NamedSharding(self.mesh, spec)
                dist_array = jax.make_array_from_single_device_arrays(
                    shape=variable.shape, sharding=out_sharding, arrays=device_buffers
                )
                
                # Wrapper
                new_var = ShardedWeight(dist_array, param_name, trainable=is_trainable)
                
                # Cleanup JAX
                try:
                    jax.block_until_ready(dist_array)
                    jax.clear_caches()
                except: pass

            else:
                # MPMD Fallback (Rank 0 only logic)
                shard_np = split_tensor_for_parallelism(
                    variable, index=self.rank, device_count=self.device_count, dim=split_dim
                )
                new_var = ShardedWeight(shard_np, param_name, trainable=is_trainable)
                del shard_np

            # 4. SWAP IN PLACE
            # Find which attribute on 'layer' points to 'variable'
            found_attr = False
            for attr_name in dir(layer):
                # Skip privates and methods
                if attr_name.startswith("__"): continue
                try:
                    val = getattr(layer, attr_name)
                    # We compare IDs to find the attribute name for this variable
                    if id(val) == id(variable):
                        setattr(layer, attr_name, new_var)
                        found_attr = True
                        # print(f"      ✅ Swapped layer.{attr_name}")
                        # Don't break, might be referenced by multiple attrs
                except:
                    pass
            
            if found_attr:
                modified_set.add(param_name)
                self.sharded_weights[param_name] = new_var
                
                # Free old memory
                try:
                    dummy = np.zeros((1,), dtype=variable.dtype)
                    variable.assign(dummy)
                except: pass
                
                gc.collect()

        except Exception as e:
            print(f"   ❌ Failed to swap {param_name}: {e}")
            raise e


def _define_parameter_sharded_model():
    from keras.src.models import Model
    
    class ParameterShardedModel(Model):
        def __init__(self, original_model, sharding_strategy, config, device_id):
            super().__init__()
            self.original_model = original_model
            self.sharding_strategy = sharding_strategy
            self.config = config
            self.device_id = device_id
            
            # Since we swapped variables in-place, we don't need to rebuild variables list manually.
            # self.original_model.variables will now point to the ShardedWeights!

        def call(self, inputs, training=None, mask=None):
            outputs = self.original_model(inputs, training=training, mask=mask)
            
            # Apply AllReduce Output Rules
            if hasattr(self.config, "output_rules"):
                for layer_name, rule in self.config.output_rules.items():
                    if rule == "allreduce sum":
                         outputs = distribution_lib.all_reduce(outputs, op="sum", axis_name="model")
            return outputs

    return ParameterShardedModel