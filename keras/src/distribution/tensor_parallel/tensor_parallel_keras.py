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
        model, # Can be a Model instance OR a Callable (Factory)
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
        
        # --- 1. MEMORY PRESERVATION: Load & Offload ---
        self.temp_dir = tempfile.mkdtemp(prefix="tp_weights_")
        
        # Handle Factory vs Instance
        if callable(model) and not isinstance(model, keras.Model):
            print("🏭 Executing Model Factory to load Master Model...")
            loaded_model = model() # Load 18GB
        else:
            loaded_model = model # Use existing (Risk of external ref!)

        # Capture Config & Class BEFORE deletion
        self.model_config = loaded_model.get_config()
        self.model_cls = loaded_model.__class__
        
        # Capture Sharding Config (requires structure inspection)
        self.tensor_parallel_config = get_default_config(loaded_model, self.devices)

        # Save Weights to Disk
        print(f"💾 Offloading {len(loaded_model.variables)} variables to disk...")
        self._save_weights_to_disk(loaded_model)
        
        # CRITICAL: Destruction
        print("🗑️  Destroying Master Model from RAM...")
        del loaded_model # Delete local reference
        if 'model' in locals(): del model # Delete argument reference
        gc.collect() # Force Reclaim 18GB
        
        # Verify Memory (Optional Check)
        print("✅ Master Model destruction complete. RAM should be free.")

        self.model_shards = []
        print(f"🚀 Initializing Tensor Parallelism on {self.devices}")

        # --- 2. Lazy Sharding Loop ---
        for rank, device_id in enumerate(self.devices):
            print(f"[{device_id}] ⏳ Creating shard {rank+1}/{self.device_count}...")
            
            # A. Create Skeleton (CPU) -> ~18GB (Allocated here, freed from Master)
            # Since Master is gone, we have room for 1 Skeleton.
            with keras.device("cpu"):
                shard = self.model_cls.from_config(self.model_config)
                # Build is tricky without inputs, but usually from_config handles layers.
                # If we need to build:
                if hasattr(shard, 'build_from_config'):
                     shard.build_from_config(self.model_config)

            # B. Stream Weights (Disk -> CPU Slice -> GPU)
            # This iteratively replaces the random weights in 'shard' with proper slices
            shard, _ = make_parameter_sharded_model(
                shard_model=shard,
                weight_loader=self._weight_loader, 
                config=self.tensor_parallel_config,
                rank=rank,
                device_count=self.device_count,
                device_id=device_id,
            )

            # C. Store Shard
            self.model_shards.append(shard)
            
            # D. Cleanup CPU Skeleton
            # 'shard' variable now holds the model which has mostly been pushed to GPU?
            # Actually, Keras variables stick to device. 
            # But the 'shard' object structure remains on CPU.
            gc.collect()
            print(f"[{device_id}] ✅ Shard ready.")

        # Cleanup Disk
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
            # Save as numpy to avoid keeping tensor graph alive
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
        inputs = {
            i.name.split(':')[0]: keras.Input(shape=i.shape[1:], dtype=i.dtype, name=i.name.split(':')[0])
            for i in ref.inputs
        }
        shard_outputs = []
        for shard in self.model_shards:
            shard_in = {
                k: inputs[k.split(':')[0]] for k in [x.name for x in shard.inputs] 
                if k.split(':')[0] in inputs
            }
            if not shard_in: shard_in = inputs
            shard_outputs.append(shard(shard_in))
        
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