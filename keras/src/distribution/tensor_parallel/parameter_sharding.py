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
    """
    Creates a Distributed Variable spanning multiple devices (SPMD) 
    OR a single-device variable (MPMD).
    """
    def __init__(self, tensor_val, name, trainable=True, device_id=None):
        
        if name:
            clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        else:
            clean_name = "sharded_var"
            
        # Case A: SPMD (JAX Distributed Array passed directly)
        if hasattr(tensor_val, 'sharding') and tensor_val.sharding is not None:
            # We do NOT use 'with device(...)' here because the array 
            # is already physically distributed across the mesh.
            self._variable = Variable(
                initializer=tensor_val, 
                trainable=trainable, 
                name=clean_name,
                dtype=tensor_val.dtype
            )
            
        # Case B: MPMD (Numpy array passed, need to push to specific device)
        else:
            current_dev = device_id if device_id else "cpu"
            with device(current_dev):
                self._variable = Variable(
                    initializer=tensor_val, 
                    trainable=trainable, 
                    name=clean_name,
                    dtype=tensor_val.dtype 
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
        
        # [SPMD DETECTOR] Check if we can see multiple GPUs locally
        self.local_devices = jax.local_devices() if jax else []
        # If we see as many devices as requested, we are in Notebook/Single-Process mode
        self.is_spmd = len(self.local_devices) >= self.device_count
        
        if self.is_spmd:
            print(f"   [Strategy] SPMD Mode Active: Managing {len(self.local_devices)} local GPUs directly.")
            # Create a 1D logical mesh: (1, N) for Model Parallelism
            self.mesh = Mesh(mesh_utils.create_device_mesh((1, self.device_count)), axis_names=('batch', 'model'))
        else:
            print(f"   [Strategy] MPMD Mode Active: Managing Rank {self.rank} only.")

    def shard_model_parameters(self, model, config, device_id: Any) -> Tuple["Model", Set[str]]:
        ParameterShardedModel = _define_parameter_sharded_model()
        
        mode_str = "SPMD (Dual-GPU)" if self.is_spmd else f"MPMD (Rank {self.rank})"
        print(f"🔧 Applying parameter sharding [v9-{mode_str}]...")
        log_memory("Start")
        
        modified_parameters = set()
        
        # 1. Count items
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
                        # -----------------------------------------------------------
                        # 1. Force Freeze Logic (Saves 36GB Optimizer RAM)
                        # -----------------------------------------------------------
                        is_trainable = False
                        if "lora" in param_name.lower():
                            is_trainable = True
                            print(f"   [{current_idx}/{total_items}] 🔥 Sharding Trainable LoRA: {param_name}")
                        
                        # -----------------------------------------------------------
                        # 2. SPMD Logic (Load BOTH GPUs)
                        # -----------------------------------------------------------
                        if self.is_spmd:
                            device_buffers = []
                            
                            # Loop through ALL devices (0, 1, ...), slice, and upload
                            for dev_idx in range(self.device_count):
                                # a. Slice on CPU
                                shard_np = split_tensor_for_parallelism(
                                    param, 
                                    index=dev_idx, 
                                    device_count=self.device_count, 
                                    dim=split_dim
                                )
                                # b. Push to specific GPU (gpu:0 then gpu:1)
                                target_dev = self.local_devices[dev_idx]
                                dev_buffer = jax.device_put(shard_np, target_dev)
                                device_buffers.append(dev_buffer)
                                
                                # c. Free CPU shard immediately
                                del shard_np
                            
                            # d. Stitch into Global Array
                            # Define how the final array is split: PartitionSpec(None, 'model') or ('model', None)
                            spec = PartitionSpec(None, 'model') if split_dim == 1 else PartitionSpec('model', None)
                            out_sharding = NamedSharding(self.mesh, spec)
                            
                            # Create the Distributed JAX Array
                            distributed_array = jax.make_array_from_single_device_arrays(
                                shape=param.shape,
                                sharding=out_sharding,
                                arrays=device_buffers
                            )
                            
                            # e. Create Wrapper
                            sharded_var = ShardedWeight(
                                distributed_array,
                                name=param_name,
                                trainable=is_trainable 
                            )
                            
                        # -----------------------------------------------------------
                        # 3. MPMD Logic (Legacy/Cluster)
                        # -----------------------------------------------------------
                        else:
                            shard_np = split_tensor_for_parallelism(
                                param, index=self.rank, device_count=self.device_count, dim=split_dim
                            )
                            sharded_var = ShardedWeight(
                                shard_np, 
                                name=param_name, 
                                trainable=is_trainable,
                                device_id=device_id
                            )
                            del shard_np

                        self.sharded_weights[param_name] = sharded_var
                        modified_parameters.add(param_name)
                        
                        # -----------------------------------------------------------
                        # 4. Cleanup
                        # -----------------------------------------------------------
                        # Release JAX fragmentation
                        if jax is not None and self.is_spmd:
                            try:
                                jax.block_until_ready(distributed_array)
                                jax.clear_caches()
                            except: pass

                        # Release Host RAM
                        try:
                            dummy = np.zeros((1,), dtype=param.dtype)
                            param.assign(dummy)
                        except: pass
                        
                        if current_idx % 50 == 0:
                            gc.collect()
                            log_memory(f"Step {current_idx}")
                        
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
            # In SPMD mode, inputs might need to be explicitly sharded before calling
            # But usually JAX handles automatic sharding propagation if weights are sharded.
            outputs = self.original_model(inputs, training=training, mask=mask)
            
            if hasattr(self.config, "output_rules"):
                for layer_name, rule in self.config.output_rules.items():
                    if rule == "allreduce sum":
                         outputs = distribution_lib.all_reduce(outputs, op="sum", axis_name="model")
            return outputs

    return ParameterShardedModel