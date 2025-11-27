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
    import jax.numpy as jnp
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
            
        # SPMD: tensor_val is already a global jax.Array
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
            # Create the mesh needed for Input Distribution
            self.mesh = Mesh(mesh_utils.create_device_mesh((1, self.device_count)), axis_names=('batch', 'model'))
        else:
            print(f"   [Strategy] MPMD Mode: Managing Rank {self.rank}.")
            self.mesh = None

    def shard_model_parameters(self, model, config, device_id: Any) -> Tuple["Model", Set[str]]:
        # Pass the mesh to the model so it can distribute inputs
        ParameterShardedModel = _define_parameter_sharded_model()
        
        print(f"🔧 Applying parameter sharding [v11-Input-Dist]...")
        log_memory("Start")
        
        modified_parameters = set()
        self._recursive_shard_layers(model, config, modified_parameters)

        sharded_model = ParameterShardedModel(
            original_model=model,
            sharding_strategy=self,
            config=config,
            device_id=device_id,
            mesh=self.mesh # <--- Pass mesh here
        )
        
        print(f"🎯 Sharding Complete. Replaced {len(modified_parameters)} parameters.")
        log_memory("Final")
        return sharded_model, modified_parameters

    def _recursive_shard_layers(self, current_layer, config, modified_set):
        if hasattr(current_layer, 'layers'):
            for sub_layer in current_layer.layers:
                self._recursive_shard_layers(sub_layer, config, modified_set)
                
        layer_vars = []
        if hasattr(current_layer, "weights"):
            layer_vars = current_layer.weights
        
        if not layer_vars: return

        for variable in layer_vars:
            rule = config.get(variable.path, config.get(variable.name, None))
            if rule is None:
                for pattern, r in config.items():
                    if isinstance(pattern, str) and (re.fullmatch(pattern, variable.path) or re.fullmatch(pattern, variable.name)):
                        rule = r
                        break
            
            if rule is not None:
                self._shard_and_swap(current_layer, variable, rule, modified_set)

    def _shard_and_swap(self, layer, variable, rule, modified_set):
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

            is_trainable = False
            if "lora" in param_name.lower():
                is_trainable = True
                print(f"      🔥 Keeping LoRA Trainable: {param_name}")

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

            # Swap in place
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
            self.mesh = mesh # JAX Mesh for inputs

        def _distribute_inputs(self, inputs):
            """
            Forces inputs to be replicated across the mesh to prevent
            JAX from gathering weights to a single GPU.
            """
            if self.mesh is None or jax is None:
                return inputs

            # Helper to put a single tensor on the mesh (Replicated)
            def _put(x):
                if not isinstance(x, (np.ndarray, jax.Array)):
                    return x
                # If already distributed, skip
                if hasattr(x, 'sharding') and x.sharding is not None:
                    return x
                
                # Replicate: We want x to be available on ALL devices.
                # Spec: (None, None) means replicated on all axes if axis_names are used, 
                # but we need explicit NamedSharding.
                # Since our mesh is (1, 2) ['batch', 'model'],
                # we want data to be replicated on 'model' axis.
                # Usually data is partitioned on 'batch' axis? 
                # Wait, our mesh size 1 on 'batch' means "All devices process full batch"??
                # No, standard TP means we split 'model' axis. Data is usually replicated.
                
                # Let's effectively replicate it everywhere.
                sharding = NamedSharding(self.mesh, PartitionSpec()) # Fully replicated
                return jax.device_put(x, sharding)

            return jax.tree.map(_put, inputs)

        def call(self, inputs, training=None, mask=None):
            # [CRITICAL] Distribute inputs before execution
            d_inputs = self._distribute_inputs(inputs)
            
            outputs = self.original_model(d_inputs, training=training, mask=mask)
            
            if hasattr(self.config, "output_rules"):
                for layer_name, rule in self.config.output_rules.items():
                    if rule == "allreduce sum":
                         outputs = distribution_lib.all_reduce(outputs, op="sum", axis_name="model")
            return outputs

    return ParameterShardedModel