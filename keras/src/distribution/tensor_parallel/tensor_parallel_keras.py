import logging
import gc
import os
import shutil
import tempfile
import numpy as np
import keras
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
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Device Setup
        if device_count is None or device_ids is None:
            all_devices = list_devices()
            device_count = len(all_devices)
            device_ids = [str(d) for d in all_devices]

        self.device_count = device_count
        self.devices = [self._normalize_device_id(d) for d in device_ids]
        
        # --- 1. MEMORY PRESERVATION ---
        self.temp_dir = tempfile.mkdtemp(prefix="tp_weights_")
        
        if callable(model) and not isinstance(model, keras.Model):
            print("🏭 Executing Model Factory to load Master Model...")
            loaded_model = model()
        else:
            loaded_model = model

        self.model_config = loaded_model.get_config()
        self.model_cls = loaded_model.__class__
        
        # Capture Input Shape for Manual Build (Fix for UserWarning)
        self.input_shapes = None
        if hasattr(loaded_model, "inputs") and loaded_model.inputs:
             self.input_shapes = [i.shape for i in loaded_model.inputs]

        self.tensor_parallel_config = get_default_config(loaded_model, self.devices)

        print(f"💾 Offloading {len(loaded_model.variables)} variables to disk...")
        self._save_weights_to_disk(loaded_model)
        
        print("🗑️  Destroying Master Model from RAM...")
        del loaded_model
        if 'model' in locals(): del model
        gc.collect()
        
        self.model_shards = []
        print(f"🚀 Initializing Tensor Parallelism on {self.devices}")

        # --- 2. Lazy Sharding Loop ---
        for rank, device_id in enumerate(self.devices):
            print(f"[{device_id}] ⏳ Creating shard {rank+1}/{self.device_count}...")
            
            # A. Create Skeleton (CPU)
            with keras.device("cpu"):
                shard = self.model_cls.from_config(self.model_config)
                
                # --- FIX 1: UNIQUE NAMING ---
                # Essential for Keras Functional API to accept multiple shards
                shard._name = f"{shard.name}_shard_{rank}"
                
                # --- FIX 2: ROBUST BUILD ---
                # If build_from_config fails (common in KerasNLP), force manual build
                if not shard.built and self.input_shapes:
                    try:
                        shard.build(self.input_shapes)
                    except Exception:
                        # Fallback: KerasHub sometimes requires specific input structures (dict)
                        pass
                
                # Attempt standard config build if still needed
                if not shard.built and hasattr(shard, 'build_from_config'):
                     try:
                        shard.build_from_config(self.model_config)
                     except Exception as e:
                        print(f"⚠️ build_from_config skipped: {e}")

            # B. Stream & Slice
            shard, _ = make_parameter_sharded_model(
                shard_model=shard,
                weight_loader=self._weight_loader, 
                config=self.tensor_parallel_config,
                rank=rank,
                device_count=self.device_count,
                device_id=device_id,
            )

            self.model_shards.append(shard)
            
            gc.collect()
            print(f"[{device_id}] ✅ Shard ready.")

        try:
            shutil.rmtree(self.temp_dir)
        except Exception:
            pass
        
        self.built = True
        self.distributed = True
        self.assembled_model = self.build_assembled_model()

    def _save_weights_to_disk(self, model):
        for v in model.variables:
            name = v.path if hasattr(v, 'path') else v.name
            safe_name = name.replace("/", "_").replace(":", "_")
            path = os.path.join(self.temp_dir, safe_name + ".npy")
            np.save(path, v.numpy())

    def _weight_loader(self, param_name):
        safe_name = param_name.replace("/", "_").replace(":", "_")
        path = os.path.join(self.temp_dir, safe_name + ".npy")
        if os.path.exists(path):
            return np.load(path)
        return None

    def _normalize_device_id(self, device_id):
        d_str = str(device_id).lower()
        if d_str.startswith("cuda:"): return d_str.replace("cuda:", "gpu:")
        if ":" not in d_str: return f"gpu:{d_str}"
        return d_str

    def build_assembled_model(self):
        ref = self.model_shards[0]
        
        # Reconstruct inputs
        inputs = {}
        # Try to detect input names from the reference shard
        if hasattr(ref, "input_names"):
             input_names = ref.input_names
        else:
             input_names = [i.name.split(':')[0] for i in ref.inputs]

        # Create symbolic inputs
        for idx, i in enumerate(ref.inputs):
            name = i.name.split(':')[0]
            inputs[name] = keras.Input(shape=i.shape[1:], dtype=i.dtype, name=name)

        shard_outputs = []
        for shard in self.model_shards:
            # Map inputs to shard
            # Handle list vs dict inputs based on model signature
            try:
                # Try dict input first (Standard for KerasNLP)
                shard_out = shard(inputs)
            except Exception:
                # Fallback to list input
                shard_out = shard(list(inputs.values()))
            
            shard_outputs.append(shard_out)
        
        if len(shard_outputs) > 1:
            out = keras.layers.Add()(shard_outputs)
        else:
            out = shard_outputs[0]
            
        return keras.Model(inputs=inputs, outputs=out)

    def call(self, inputs, training=None, **kwargs):
        return self.assembled_model(inputs, training=training, **kwargs)

    def compile(self, optimizer=None, **kwargs):
        if len(self.model_shards) > 1 and optimizer:
            opt = TensorParallelOptimizer(optimizer, self.device_count)
            opt._shard_models = self.model_shards
            var_map = {}
            for i, shard in enumerate(self.model_shards):
                for v in shard.trainable_variables:
                    key = v.path if hasattr(v, "path") else v.name
                    if key not in var_map: var_map[key] = [None]*self.device_count
                    var_map[key][i] = v
            opt._shard_var_map = var_map
            super().compile(optimizer=opt, **kwargs)
        else:
            super().compile(optimizer=optimizer, **kwargs)