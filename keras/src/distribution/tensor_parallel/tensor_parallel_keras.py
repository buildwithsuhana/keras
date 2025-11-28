import logging
import gc
import os
import shutil
import tempfile
import ctypes
from typing import Optional, Sequence, Union

import numpy as np
import keras
from keras import ops
from keras.src.distribution.tensor_parallel.autoconfig import get_default_config
from keras.src.distribution.tensor_parallel.parameter_sharding import make_parameter_sharded_model
from keras.src.distribution.tensor_parallel.coordinated_optimizer import TensorParallelOptimizer
from keras.src.distribution import list_devices
from keras.src.models import Model

logger = logging.getLogger(__file__)

class TensorParallelKeras(Model):
    def __init__(
        self,
        model,
        device_count=None,
        device_ids=None,
        low_memory_mode=True, 
        **kwargs,
    ):
        super().__init__(**kwargs)

        if device_count is None:
            device_count, device_ids = self._auto_detect_parallelism()
        elif device_ids is None:
            device_ids = self._auto_configure_devices(device_count)

        self.device_count = device_count
        self.devices = list(self.check_device_ids(device_ids))
        if len(self.devices) != device_count:
            self.devices = self._adjust_device_list(self.devices, device_count)
            
        self.distributed = self.device_count > 1
        
        # 1. Config Generation
        device_names = [str(d) for d in self.devices]
        self.tensor_parallel_config = get_default_config(model, device_names)

        self.model_shards = []
        self.modified_parameters_names = set()

        if not self.distributed:
            self.model_shards = [model]
            self.built = True
            self.assembled_model = model
            return

        print(f"🔧 Sharding {model.name} across {len(self.devices)} devices: {self.devices}")
        
        # 2. ZERO STAGE INIT (Low Memory Mode)
        temp_dir = None
        weights_path = None
        
        # Cache metadata & config
        self._input_specs = [{'shape': i.shape, 'dtype': i.dtype, 'name': i.name} for i in model.inputs]
        self._original_name = model.name
        self._model_config = model.get_config() # Save config to rebuild skeleton
        
        if low_memory_mode:
            print("📉 Low Memory Mode: Enabled")
            temp_dir = tempfile.mkdtemp()
            weights_path = os.path.join(temp_dir, "temp_weights.weights.h5")
            try:
                # A. Save ONLY weights (cheaper/safer than full model)
                model.save_weights(weights_path)
                print(f"   💾 Weights offloaded to {weights_path}")
                
                # B. Aggressive Memory Cleanup
                print("   🧹 Aggressively freeing CPU memory...")
                self._force_release_memory(model)
                del model
                
            except Exception as e:
                logger.warning(f"Failed to offload model: {e}. Fallback to clone.")
                weights_path = None

        # 3. Sharding Loop
        for rank, device_id in enumerate(self.devices):
            print(f"[{device_id}] ➡️  Processing Rank {rank}")
            self._force_gc() # Pre-emptive GC
            
            # A. Create Skeleton & Load Weights (CPU)
            replica_model = None
            if low_memory_mode and weights_path:
                with keras.saving.custom_object_scope({}): # Add custom objects if needed
                    with keras.device("cpu"):
                        try:
                            # 1. Rebuild empty shell (Cheap)
                            # We use the class of the original model if possible, or Functional/Sequential
                            # But since we have config, from_config is best.
                            # For KerasNLP models, we might need the specific class.
                            # We'll try generic reconstruction which works for Functional/Sequential
                            try:
                                replica_model = self.__class__.from_config(self._model_config)
                            except:
                                # Fallback: Clone the 'model' variable if it wasn't deleted? 
                                # We deleted it. So we rely on Keras functional reconstruction.
                                # If 'model' was a custom class, from_config on base Model might fail.
                                # KerasNLP models usually support from_config.
                                from keras.src.models import Model as BaseKerasModel
                                replica_model = BaseKerasModel.from_config(self._model_config)

                            # 2. Load Weights (Heavy, peaks at 18GB)
                            replica_model.load_weights(weights_path)
                            
                        except Exception as e:
                            logger.error(f"Error rebuilding replica: {e}")
                            raise e
            else:
                # Fallback path (High Memory)
                try:
                    replica_model = keras.models.clone_model(model)
                    replica_model.set_weights(model.get_weights())
                except:
                    replica_model = model

            # B. Shard (Move slice to GPU)
            shard, modified_names = make_parameter_sharded_model(
                replica_model,
                self.tensor_parallel_config,
                rank=rank,
                device_count=self.device_count,
                device_id=device_id,
            )
            self.model_shards.append(shard)
            self.modified_parameters_names.update(modified_names)
            
            # C. Cleanup Replica
            # Detach anything we can
            self._force_release_memory(replica_model)
            del replica_model 
            self._force_gc()

        # 4. Cleanup Disk
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

        self.built = True
        self.assembled_model = self.build_assembled_model()

    def _force_release_memory(self, model):
        """Destroys model weights and forces OS memory release."""
        if not model: return
        
        # 1. Iterate all layers (including nested)
        # _flatten_layers is a robust internal Keras method if available
        layers = getattr(model, '_flatten_layers', None)
        if callable(layers):
            all_layers = layers(include_self=True, recursive=True)
        else:
            # Fallback manual recursion
            all_layers = []
            def recurse(m):
                all_layers.append(m)
                if hasattr(m, 'layers'):
                    for l in m.layers: recurse(l)
            recurse(model)

        # 2. Nullify weights
        for layer in all_layers:
            # Clear public weight lists
            if hasattr(layer, '_trainable_weights'): layer._trainable_weights = []
            if hasattr(layer, '_non_trainable_weights'): layer._non_trainable_weights = []
            # Clear standard attributes
            for attr in ['kernel', 'bias', 'embeddings', 'variable', 'moving_mean', 'moving_variance']:
                if hasattr(layer, attr):
                    try: setattr(layer, attr, None)
                    except: pass
        
        # 3. Clear Model cache
        if hasattr(model, '_trainable_variables'): model._trainable_variables = []
        
        # 4. FORCE SYSTEM GC
        self._force_gc()

    def _force_gc(self):
        gc.collect()
        try:
            # Linux only: Trim malloc heap to release memory back to OS
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except:
            pass

    @property
    def variables(self):
        unique_vars = {id(var): var for shard in self.model_shards for var in shard.variables}
        return list(unique_vars.values())

    @property
    def trainable_variables(self):
        unique_vars = {id(var): var for shard in self.model_shards for var in shard.trainable_variables}
        return list(unique_vars.values())

    def _auto_detect_parallelism(self):
        from keras.src.distribution import get_best_devices
        available_devices = list_devices()
        count = len(available_devices)
        return count, get_best_devices(count)

    def _adjust_device_list(self, device_ids, target_count):
        current = len(device_ids)
        if current >= target_count: return device_ids[:target_count]
        return list(device_ids) + [f"cpu:{i}" for i in range(current, target_count)]

    def _auto_configure_devices(self, count):
        devs = list_devices()
        return devs[:count] if devs else ["cpu:0"]

    def check_device_ids(self, ids):
        if ids is None: ids = list_devices()
        return tuple(self.canonicalize_device(d) for d in ids)

    def canonicalize_device(self, device_spec: Union[str, int, any]) -> str:
        if hasattr(device_spec, 'id') and hasattr(device_spec, 'platform'):
             return f"{device_spec.platform}:{device_spec.id}"
        s_device = str(device_spec).lower()
        if "gpu" in s_device or "cuda" in s_device:
            import re
            match = re.search(r'\d+', s_device)
            idx = match.group() if match else "0"
            return f"gpu:{idx}"
        elif "tpu" in s_device:
            import re
            match = re.search(r'\d+', s_device)
            idx = match.group() if match else "0"
            return f"tpu:{idx}"
        if isinstance(device_spec, int):
            if device_spec == -1: return "cpu"
            return f"gpu:{device_spec}"
        return "cpu"

    def build_assembled_model(self):
        if not self.distributed: return self.model_shards[0]
        input_layers = {}
        for spec in self._input_specs:
            clean = spec['name'].split(":")[0]
            input_layers[clean] = keras.Input(shape=spec['shape'][1:], dtype=spec['dtype'], name=clean)
        partial_outputs = []
        for shard in self.model_shards:
            shard_inputs = dict(input_layers)
            partial_outputs.append(shard(shard_inputs))
        return keras.Model(inputs=list(input_layers.values()), outputs=partial_outputs[0])

    def call(self, inputs, training=None, **kwargs):
        return self.assembled_model(inputs, training=training, **kwargs)

    def compile(self, optimizer=None, loss=None, metrics=None, **kwargs):
        if len(self.model_shards) > 1 and optimizer is not None:
            self.coordinated_optimizer = TensorParallelOptimizer(
                optimizer, self.device_count, self.tensor_parallel_config
            )
            self.coordinated_optimizer._shard_models = self.model_shards
            var_map = {}
            assembled = getattr(self, "assembled_model", None)
            if assembled:
                for a_var in assembled.variables:
                    key = a_var.path if hasattr(a_var, 'path') else a_var.name
                    suffix = key.split("/")[-1]
                    per_shard = []
                    for shard in self.model_shards:
                        found = next((v for v in shard.variables if v.name.endswith(suffix)), None)
                        per_shard.append(found)
                    var_map[key] = per_shard
            self.coordinated_optimizer._shard_var_map = var_map
            super().compile(optimizer=self.coordinated_optimizer, loss=loss, metrics=metrics, **kwargs)
        else:
            super().compile(optimizer, loss, metrics, **kwargs)