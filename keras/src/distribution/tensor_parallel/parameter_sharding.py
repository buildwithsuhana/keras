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
        
        print(f"🔧 Applying parameter sharding [v14-Stack-Traversal]...")
        log_memory("Start")
        
        modified_parameters = set()
        
        # USE ITERATIVE STACK (Like autoconfig)
        self._iterative_shard_layers(model, config, modified_parameters)

        sharded_model = ParameterShardedModel(
            original_model=model,
            sharding_strategy=self,
            config=config,
            device_id=device_id,
            mesh=self.mesh
        )
        
        print(f"🎯 Sharding Complete. Replaced {len(modified_parameters)} parameters.")
        log_memory("Final")
        
        return sharded_model, modified_parameters

    def _iterative_shard_layers(self, root_layer, config, modified_set):
        """
        Iterative traversal matching autoconfig.py logic exactly.
        """
        stack = [root_layer]
        visited = set()
        
        print("   [Traversal] Starting graph walk...")

        while stack:
            layer = stack.pop()
            if id(layer) in visited:
                continue
            visited.add(id(layer))

            # 1. Attempt to shard weights in THIS layer
            self._shard_layer_weights(layer, config, modified_set)

            # 2. Find children to add to stack
            children_to_add = []
            
            # Standard Keras tracking
            if hasattr(layer, 'layers') and layer.layers:
                children_to_add.extend(layer.layers)

            # Explicit Backbone attributes (Gemma/Llama)
            for specific_attr in ['token_embedding', 'embeddings', 'position_embedding', 'backbone', 'model', 'transformer_layers', 'layers', 'blocks']:
                if hasattr(layer, specific_attr):
                    attr_val = getattr(layer, specific_attr)
                    if isinstance(attr_val, layers.Layer):
                        children_to_add.append(attr_val)
                    elif isinstance(attr_val, (list, tuple)):
                        for item in attr_val:
                            if isinstance(item, layers.Layer):
                                children_to_add.append(item)

            # Aggressive Scan (fall back)
            for attr_name in dir(layer):
                if attr_name.startswith("__") or attr_name.startswith("_"): continue
                try:
                    val = getattr(layer, attr_name)
                    if isinstance(val, layers.Layer):
                        children_to_add.append(val)
                    elif isinstance(val, (list, tuple)):
                        for item in val:
                            if isinstance(item, layers.Layer):
                                children_to_add.append(item)
                except:
                    continue
            
            # Add to stack (reversed to maintain order)
            stack.extend(reversed(children_to_add))

    def _shard_layer_weights(self, layer, config, modified_set):
        if not hasattr(layer, "weights"): return
        
        for variable in layer.weights:
            # Check Config
            rule = config.get(variable.path, config.get(variable.name, None))
            if rule is None:
                for pattern, r in config.items():
                    if isinstance(pattern, str):
                        if re.fullmatch(pattern, variable.path) or re.fullmatch(pattern, variable.name):
                            rule = r; break
                        if variable.name.endswith(pattern) or variable.path.endswith(pattern):
                             rule = r; break
            
            if rule is not None:
                # We found a rule for this variable!
                self._apply_swap(layer, variable, rule, modified_set)

    def _apply_swap(self, layer, variable, rule, modified_set):
        param_name = variable.name
        if param_name in modified_set: return

        try:
            # 1. Interpret Rule
            split_dim = None
            if isinstance(rule, (tuple, list)):
                for i, axis in enumerate(rule):
                    if axis == "model": split_dim = i; break
            elif hasattr(rule, "axes"):
                for i, axis in enumerate(rule.axes):
                    if axis == "model": split_dim = i; break

            if split_dim is None: return

            # 2. Config & Create Shard
            is_trainable = False
            if "lora" in param_name.lower():
                is_trainable = True
            
            if self.is_spmd:
                device_buffers = []
                for dev_idx in range(self.device_count):
                    shard_np = split_tensor_for_parallelism(variable, index=dev_idx, device_count=self.device_count, dim=split_dim)
                    dev_buffer = jax.device_put(shard_np, self.local_devices[dev_idx])
                    device_buffers.append(dev_buffer)
                    del shard_np
                
                spec = PartitionSpec(None, 'model') if split_dim == 1 else PartitionSpec('model', None)
                out_sharding = NamedSharding(self.mesh, spec)
                dist_array = jax.make_array_from_single_device_arrays(variable.shape, out_sharding, device_buffers)
                new_var = ShardedWeight(dist_array, param_name, trainable=is_trainable)
                try: jax.block_until_ready(dist_array); jax.clear_caches()
                except: pass
            else:
                shard_np = split_tensor_for_parallelism(variable, index=self.rank, device_count=self.device_count, dim=split_dim)
                new_var = ShardedWeight(shard_np, param_name, trainable=is_trainable)
                del shard_np

            # 3. SWAP LOGIC
            swapped = False
            
            # Strategy A: Public Attributes
            for attr_name in dir(layer):
                if attr_name.startswith("__"): continue
                try:
                    if id(getattr(layer, attr_name)) == id(variable):
                        setattr(layer, attr_name, new_var)
                        swapped = True
                        # print(f"      [DEBUG] Swapped layer.{attr_name}")
                except: pass
            
            # Strategy B: Private Attributes (common in KerasNLP)
            if not swapped:
                for attr_name in ['_kernel', '_bias', 'kernel', 'bias', 'embeddings', 'variable']:
                    if hasattr(layer, attr_name):
                        try:
                            if id(getattr(layer, attr_name)) == id(variable):
                                setattr(layer, attr_name, new_var)
                                swapped = True
                                # print(f"      [DEBUG] Swapped layer.{attr_name} (Direct)")
                        except: pass

            # Strategy C: List Injection (Nuclear Option)
            # If attribute swap failed, we inject into _trainable_weights list
            # This ensures Keras sees the new variable during training even if the attribute isn't updated
            if not swapped:
                # This is risky but necessary if attributes are hidden
                if variable in layer._trainable_weights:
                    idx = layer._trainable_weights.index(variable)
                    layer._trainable_weights[idx] = new_var._variable
                    swapped = True
                    # print(f"      [DEBUG] Swapped via _trainable_weights list")
                elif variable in layer._non_trainable_weights:
                    idx = layer._non_trainable_weights.index(variable)
                    layer._non_trainable_weights[idx] = new_var._variable
                    swapped = True

            if swapped:
                modified_set.add(param_name)
                self.sharded_weights[param_name] = new_var
                try:
                    dummy = np.zeros((1,), dtype=variable.dtype)
                    variable.assign(dummy)
                except: pass
                gc.collect()
                
                if len(modified_set) % 20 == 0:
                    print(f"   [Swap] {len(modified_set)} params replaced...")
            else:
                print(f"   ⚠️ [WARN] Could not find attribute for {param_name} in {layer.name}")

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