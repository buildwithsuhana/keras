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
    def __init__(self, tensor_val, name, trainable=True):
        if name:
            clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        else:
            clean_name = "sharded_var"
            
        self._variable = Variable(
            initializer=tensor_val, 
            trainable=trainable, 
            name=clean_name,
            dtype=tensor_val.dtype
        )

    def __getattr__(self, name):
        return getattr(self._variable, name)

class ParameterShardingStrategy:
    def __init__(self, device_count: int, rank: int):
        self.device_count = device_count
        self.rank = rank
        self.sharded_weights = {} 
        
        self.local_devices = jax.local_devices() if jax else []
        self.is_spmd = len(self.local_devices) >= self.device_count
        
        if self.is_spmd:
            print(f"   [Strategy] SPMD Mode: Managing {len(self.local_devices)} local GPUs.")
            self.mesh = Mesh(mesh_utils.create_device_mesh((1, self.device_count)), axis_names=('batch', 'model'))
        else:
            print(f"   [Strategy] MPMD Mode: Managing Rank {self.rank}.")
            self.mesh = None

    def shard_model_parameters(self, model, config, device_id: Any) -> Tuple["Model", Set[str]]:
        ParameterShardedModel = _define_parameter_sharded_model()
        
        print(f"🔧 Applying parameter sharding [v13-Aggressive-Traversal]...")
        log_memory("Start")
        
        modified_parameters = set()
        visited_layers = set()
        
        # Start robust recursion
        self._recursive_shard_layers(model, config, modified_parameters, visited_layers)

        sharded_model = ParameterShardedModel(
            original_model=model,
            sharding_strategy=self,
            config=config,
            device_id=device_id,
            mesh=self.mesh
        )
        
        print(f"🎯 Sharding Complete. Replaced {len(modified_parameters)} parameters.")
        log_memory("Final")
        
        if len(modified_parameters) < 10:
            print("⚠️ WARNING: Extremely low replacement count! Model is likely still on CPU/GPU0.")
            
        return sharded_model, modified_parameters

    def _recursive_shard_layers(self, current_layer, config, modified_set, visited_layers):
        """
        Recursively walks layers and swaps weights.
        """
        # Prevent infinite recursion / cycles
        if id(current_layer) in visited_layers:
            return
        visited_layers.add(id(current_layer))

        # 1. Process weights of THIS layer
        self._shard_and_swap_layer_weights(current_layer, config, modified_set)

        # 2. Find children via standard API (Keras tracking)
        if hasattr(current_layer, 'layers'):
            for sub_layer in current_layer.layers:
                self._recursive_shard_layers(sub_layer, config, modified_set, visited_layers)

        # 3. Aggressive Attribute Scanning (Finds untracked/private sublayers)
        # [FIX] Removed startswith("_") check to find private layers
        for attr_name in dir(current_layer):
            if attr_name.startswith("__"): continue 
            
            try:
                val = getattr(current_layer, attr_name)
            except:
                continue
            
            if isinstance(val, layers.Layer):
                self._recursive_shard_layers(val, config, modified_set, visited_layers)
            elif isinstance(val, (list, tuple)):
                for item in val:
                    if isinstance(item, layers.Layer):
                        self._recursive_shard_layers(item, config, modified_set, visited_layers)

    def _shard_and_swap_layer_weights(self, layer, config, modified_set):
        if not hasattr(layer, "weights"): return
        
        for variable in layer.weights:
            # Try exact path match first
            rule = config.get(variable.path, config.get(variable.name, None))
            
            # Fuzzy/Regex match fallback
            if rule is None:
                for pattern, r in config.items():
                    if isinstance(pattern, str):
                        if re.fullmatch(pattern, variable.path) or re.fullmatch(pattern, variable.name):
                            rule = r
                            break
                        # Allow partial endswith match for robustness (e.g. "kernel" matching "dense/kernel")
                        if variable.name.endswith(pattern) or variable.path.endswith(pattern):
                             rule = r
                             break
            
            if rule is not None:
                self._apply_swap(layer, variable, rule, modified_set)

    def _apply_swap(self, layer, variable, rule, modified_set):
        param_name = variable.name
        if param_name in modified_set: return

        try:
            split_dim = None
            if isinstance(rule, (tuple, list)):
                for i, axis in enumerate(rule):
                    if axis == "model": split_dim = i; break
            elif hasattr(rule, "axes"):
                for i, axis in enumerate(rule.axes):
                    if axis == "model": split_dim = i; break

            if split_dim is None: return

            # Force Freeze Check
            is_trainable = False
            if "lora" in param_name.lower():
                is_trainable = True
                print(f"      🔥 Keeping LoRA Trainable: {param_name}")

            # SPMD Creation
            if self.is_spmd:
                device_buffers = []
                for dev_idx in range(self.device_count):
                    shard_np = split_tensor_for_parallelism(
                        variable, index=dev_idx, device_count=self.device_count, dim=split_dim
                    )
                    dev_buffer = jax.device_put(shard_np, self.local_devices[dev_idx])
                    device_buffers.append(dev_buffer)
                    del shard_np
                
                spec = PartitionSpec(None, 'model') if split_dim == 1 else PartitionSpec('model', None)
                out_sharding = NamedSharding(self.mesh, spec)
                dist_array = jax.make_array_from_single_device_arrays(
                    shape=variable.shape, sharding=out_sharding, arrays=device_buffers
                )
                
                new_var = ShardedWeight(dist_array, param_name, trainable=is_trainable)
                try:
                    jax.block_until_ready(dist_array)
                    jax.clear_caches()
                except: pass

            else:
                shard_np = split_tensor_for_parallelism(
                    variable, index=self.rank, device_count=self.device_count, dim=split_dim
                )
                new_var = ShardedWeight(shard_np, param_name, trainable=is_trainable)
                del shard_np

            # In-Place Attribute Swap
            # This is the magic that actually replaces the weight in the model
            found = False
            for attr_name in dir(layer):
                if attr_name.startswith("__"): continue
                try:
                    if id(getattr(layer, attr_name)) == id(variable):
                        setattr(layer, attr_name, new_var)
                        found = True
                except: pass
            
            if found:
                modified_set.add(param_name)
                self.sharded_weights[param_name] = new_var
                try:
                    dummy = np.zeros((1,), dtype=variable.dtype)
                    variable.assign(dummy)
                except: pass
                gc.collect()
                
                if len(modified_set) % 20 == 0:
                    print(f"   [Swap] {len(modified_set)} params replaced...")

        except Exception as e:
            print(f"   ❌ Failed to swap {param_name}: {e}")
            raise e


def _define_parameter_sharded_model():
    from keras.src.models import Model
    
    class ParameterShardedModel(Model):
        def __init__(self, original_model, sharding_strategy, config, device_id, mesh=None):
            super().__init__()
            self.original_model = original_model
            self.sharding_strategy = sharding_strategy
            self.config = config
            self.device_id = device_id
            self.mesh = mesh

        def _distribute_inputs(self, inputs):
            if self.mesh is None or jax is None: return inputs
            def _put(x):
                if not isinstance(x, (np.ndarray, jax.Array)): return x
                if hasattr(x, 'sharding') and x.sharding is not None: return x
                sharding = NamedSharding(self.mesh, PartitionSpec()) 
                return jax.device_put(x, sharding)
            return jax.tree.map(_put, inputs)

        def call(self, inputs, training=None, mask=None):
            d_inputs = self._distribute_inputs(inputs)
            outputs = self.original_model(d_inputs, training=training, mask=mask)
            
            if hasattr(self.config, "output_rules"):
                for layer_name, rule in self.config.output_rules.items():
                    if rule == "allreduce sum":
                         outputs = distribution_lib.all_reduce(outputs, op="sum", axis_name="model")
            return outputs

    return ParameterShardedModel