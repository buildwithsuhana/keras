import logging
import gc
import os
import shutil
import tempfile
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
        model_path = None
        
        # Cache metadata
        self._input_specs = [{'shape': i.shape, 'dtype': i.dtype, 'name': i.name} for i in model.inputs]
        self._original_name = model.name
        self._last_layer_name = model.layers[-1].name

        if low_memory_mode:
            print("📉 Low Memory Mode: Enabled (Offloading to disk)")
            temp_dir = tempfile.mkdtemp()
            model_path = os.path.join(temp_dir, "temp_tp_model.keras")
            try:
                # Save full model to disk
                model.save(model_path)
                print(f"   💾 Model offloaded to {model_path}")
                
                # CRITICAL: Recursively destroy the original model in memory
                print("   🧹 Aggressively freeing CPU memory from original model...")
                self._recursively_hollow_model(model)
                
                # Force delete the reference and GC
                del model
                gc.collect()
                
            except Exception as e:
                logger.warning(f"Failed to offload model: {e}. Using standard clone.")
                model_path = None

        # 3. Sharding Loop
        for rank, device_id in enumerate(self.devices):
            print(f"[{device_id}] ➡️  Processing Rank {rank}")
            gc.collect()
            
            # A. Revive Replica (CPU)
            replica_model = None
            if low_memory_mode and model_path:
                with keras.saving.custom_object_scope({}):
                    with keras.device("cpu"):
                        # Load strictly on CPU
                        try:
                            replica_model = keras.models.load_model(model_path, compile=False)
                        except Exception as e:
                            raise RuntimeError(f"OOM or Error loading replica for Rank {rank}: {e}")
            else:
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
            del replica_model 
            gc.collect()

        # 4. Cleanup Disk
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

        self.built = True
        self.assembled_model = self.build_assembled_model()

    def _recursively_hollow_model(self, layer_or_model):
        """
        Recursively traverses the model structure to explicitly destroy 
        variables and weights, freeing memory even if references exist.
        """
        # 1. Recurse into sub-modules (e.g. Backbone, TransformerBlock)
        if hasattr(layer_or_model, 'layers'):
            for sub_layer in layer_or_model.layers:
                self._recursively_hollow_model(sub_layer)
        
        # Also check for 'backbone' or 'token_embedding' direct attributes
        # which are common in KerasNLP but might not be in .layers list immediately
        for attr in ['backbone', 'token_embedding', 'embeddings', 'encoder', 'decoder']:
            if hasattr(layer_or_model, attr):
                val = getattr(layer_or_model, attr)
                if isinstance(val, keras.layers.Layer) or isinstance(val, keras.Model):
                    self._recursively_hollow_model(val)

        # 2. Destroy Weights on the current object
        # We iterate known weight attributes and force them to None
        for attr in ['kernel', 'bias', 'embeddings', 'gamma', 'beta', 'moving_mean', 'moving_variance', 'variable']:
            if hasattr(layer_or_model, attr):
                try:
                    setattr(layer_or_model, attr, None)
                except: pass

        # 3. Clear storage lists
        if hasattr(layer_or_model, '_trainable_weights'):
            layer_or_model._trainable_weights = []
        if hasattr(layer_or_model, '_non_trainable_weights'):
            layer_or_model._non_trainable_weights = []
        if hasattr(layer_or_model, '_trainable_variables'):
            layer_or_model._trainable_variables = []
        if hasattr(layer_or_model, '_non_trainable_variables'):
            layer_or_model._non_trainable_variables = []

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