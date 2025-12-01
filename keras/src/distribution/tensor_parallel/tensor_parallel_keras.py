import logging
import gc
import os
import re
import shutil
import tempfile
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
        
        self.temp_dir = tempfile.mkdtemp(prefix="tp_weights_")
        
        if callable(model) and not isinstance(model, keras.Model):
            print("🏭 Executing Model Factory...")
            loaded_model = model()
        else:
            loaded_model = model

        self.model_config = loaded_model.get_config()
        self.model_cls = loaded_model.__class__
        self.tensor_parallel_config = get_default_config(loaded_model, self.devices)

        print(f"💾 Offloading {len(loaded_model.variables)} variables...")
        self._save_weights_to_disk(loaded_model)
        
        print("🗑️  Destroying Master Model...")
        del loaded_model
        if 'model' in locals(): del model
        gc.collect()

        self.model_shards = []
        print(f"🚀 Initializing TP on {self.devices}")

        for rank, device_id in enumerate(self.devices):
            print(f"[{device_id}] ⏳ Creating shard {rank+1}/{self.device_count}...")
            
            # 1. Create on CPU (Skeleton)
            with keras.device("cpu"):
                shard = self.model_cls.from_config(self.model_config)
                if hasattr(shard, 'build_from_config'):
                     shard.build_from_config(self.model_config)

            # 2. Slice Weights (Disk -> CPU -> TPU)
            # Capture the set of variables that were successfully sharded
            shard, modified_vars = make_parameter_sharded_model(
                shard_model=shard,
                weight_loader=self._weight_loader, 
                config=self.tensor_parallel_config,
                rank=rank,
                device_count=self.device_count,
                device_id=device_id,
            )

            # 3. Smart Migration (Unsharded variables -> TPU)
            try:
                import jax
                target_idx = int(device_id.split(":")[-1])
                target_device = next(d for d in jax.devices() if d.id == target_idx)
                
                migrated_count = 0
                for v in shard.variables:
                    # Skip variables that are already on device (the sharded ones)
                    # We check path/name against the modified set
                    v_name = v.path if hasattr(v, 'path') else v.name
                    
                    if v_name in modified_vars:
                        continue 

                    # DIAGNOSTIC: Check for huge unsharded variables (Likely Config Mismatch)
                    # 50MB threshold
                    if hasattr(v, 'value'):
                        size_bytes = v.value.nbytes
                        if size_bytes > 50_000_000:
                            print(f"⚠️  [Rank {rank}] CRITICAL: Large variable '{v_name}' ({size_bytes/1e6:.1f}MB) was NOT sharded! Moving full size to TPU (Risk of OOM).")

                    # Move to TPU
                    v.assign(jax.device_put(v.value, target_device))
                    migrated_count += 1
                
                # print(f"   ↳ Migrated {migrated_count} small variables.")

            except Exception as e:
                print(f"⚠️ Migration Error: {e}")

            suffix = f"_tp{rank}"
            try: shard._name = shard.name + suffix
            except: pass
            for layer in getattr(shard, 'layers', []):
                try: 
                    if not layer.name.endswith(suffix): layer._name = layer.name + suffix
                except: continue

            self.model_shards.append(shard)
            gc.collect()
            print(f"[{device_id}] ✅ Shard ready.")

        try: shutil.rmtree(self.temp_dir)
        except: pass
        self.built = True

    def call(self, inputs, training=None, **kwargs):
        results = []
        for i, shard in enumerate(self.model_shards):
            target_device = self.devices[i]
            with keras.device(target_device):
                out = shard(inputs, training=training, **kwargs)
                results.append(out)
        
        total = results[0]
        for i in range(1, len(results)):
            total = ops.add(total, results[i])
        return total

    def _save_weights_to_disk(self, model):
        for v in model.variables:
            name = v.path if hasattr(v, 'path') else v.name
            safe_name = name.replace("/", "_").replace(":", "_")
            path = os.path.join(self.temp_dir, safe_name + ".npy")
            np.save(path, v.numpy())

    def _weight_loader(self, param_name):
        safe_name = param_name.replace("/", "_").replace(":", "_")
        path = os.path.join(self.temp_dir, safe_name + ".npy")
        if os.path.exists(path): return np.load(path)
        return None

    def _normalize_device_id(self, device_id):
        d_str = str(device_id).lower()
        if "tpu" in d_str and "gpu" not in d_str:
            if d_str.startswith("tpu:"): return d_str
            match = re.search(r"tpu_(\d+)", d_str)
            if match: return f"tpu:{match.group(1)}"
            if ":" not in d_str: return f"tpu:{d_str}"
            return d_str
        if d_str.startswith("cuda:"): return d_str.replace("cuda:", "gpu:")
        if ":" not in d_str: return f"gpu:{d_str}"
        return d_str

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
            
            for i, shard in enumerate(self.model_shards):
                if isinstance(optimizer, str):
                    shard_opt = keras.optimizers.get(optimizer)
                else:
                    shard_opt = optimizer.from_config(optimizer.get_config())
                
                with keras.device(self.devices[i]):
                    shard.compile(optimizer=shard_opt, **kwargs)
        else:
            super().compile(optimizer=optimizer, **kwargs)