import logging
import re
import gc
import os
import shutil
import tempfile
from typing import Collection, Optional, Sequence, Union

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
        low_memory_mode=True, # Default to True for large models
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
        
        # 1. Config Generation (Before modifying model)
        self.tensor_parallel_config = None
        device_names = [str(d) for d in self.devices]
        self.tensor_parallel_config = get_default_config(model, device_names)

        self.model_shards = []
        self.modified_parameters_names = set()

        if not self.distributed:
            self.model_shards = [model]
            self.built = True
            self.assembled_model = model
            return

        print(f"🔧 Creating Shards for {model.name} across {len(self.devices)} devices")
        
        # 2. ZERO STAGE INIT (Low Memory Mode)
        # Strategy: Save Original to Disk -> Delete Original -> Load Replica 1 -> Shard -> Delete -> Load Replica 2...
        temp_dir = None
        model_path = None
        
        # Cache metadata before deletion
        self._input_specs = [{'shape': i.shape, 'dtype': i.dtype, 'name': i.name} for i in model.inputs]
        self._original_name = model.name
        self._last_layer_name = model.layers[-1].name
        self._last_layer_config = model.layers[-1].get_config()

        if low_memory_mode:
            print("📉 Low Memory Mode: Enabled (Offloading to disk)")
            temp_dir = tempfile.mkdtemp()
            model_path = os.path.join(temp_dir, "temp_tp_model.keras")
            try:
                model.save(model_path)
                print(f"   💾 Model offloaded to {model_path}")
                del model # CRITICAL: Free 18GB/36GB from RAM
                gc.collect()
            except Exception as e:
                logger.warning(f"Failed to offload model: {e}. Using standard clone.")
                model_path = None

        # 3. Sharding Loop
        for rank, device_id in enumerate(self.devices):
            print(f"[{device_id}] ➡️  Processing Rank {rank}")
            gc.collect() # Ensure previous shard debris is gone
            
            # A. Revive Replica (CPU)
            if low_memory_mode and model_path:
                with keras.saving.custom_object_scope({}):
                    # Load strictly on CPU
                    with keras.device("cpu"):
                        replica_model = keras.models.load_model(model_path, compile=False)
            else:
                # Fallback for small models
                try:
                    replica_model = keras.models.clone_model(model)
                    replica_model.set_weights(model.get_weights())
                except:
                    replica_model = model

            # B. Shard (Move slice to GPU)
            # parameter_sharding.py handles the slicing and placement
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
            # The 'shard' wrapper retains the model, but we want to ensure 
            # no OTHER references exist so python can free the CPU weights.
            del replica_model 
            gc.collect()

        # 4. Cleanup Disk
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

        self.built = True
        self.assembled_model = self.build_assembled_model()

    # ... [Include all other standard methods: variables, compile, call, etc.] ...
    # (Copy properties and helper methods from previous correct file versions)

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

    def canonicalize_device(self, device_spec: Union[str, int]) -> str:
        if isinstance(device_spec, int):
            return "cpu" if device_spec == -1 else f"gpu:{device_spec}"
        elif isinstance(device_spec, str):
            if device_spec == "cpu": return "cpu"
            if device_spec.startswith("gpu:") or device_spec.startswith("tpu:"): return device_spec
            if device_spec.startswith("cuda:"): return f"gpu:{device_spec.split(':')[1]}"
            return device_spec
        return "cpu"

    def build_assembled_model(self):
        if not self.distributed: return self.model_shards[0]
        
        # Reconstruct inputs from metadata
        input_layers = {}
        for spec in self._input_specs:
            clean = spec['name'].split(":")[0]
            input_layers[clean] = keras.Input(shape=spec['shape'][1:], dtype=spec['dtype'], name=clean)
            
        partial_outputs = []
        for shard in self.model_shards:
            shard_inputs = dict(input_layers)
            partial_outputs.append(shard(shard_inputs))
            
        # Naive aggregation (TP usually ends with AllReduce inside the shard, so outputs are same)
        final_output = partial_outputs[0]
        
        return keras.Model(inputs=list(input_layers.values()), outputs=final_output)

    def call(self, inputs, training=None, **kwargs):
        return self.assembled_model(inputs, training=training, **kwargs)

    def compile(self, optimizer=None, loss=None, metrics=None, **kwargs):
        if len(self.model_shards) > 1 and optimizer is not None:
            self.coordinated_optimizer = TensorParallelOptimizer(
                optimizer, self.device_count, self.tensor_parallel_config
            )
            self.coordinated_optimizer._shard_models = self.model_shards
            
            # Var Mapping Logic
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