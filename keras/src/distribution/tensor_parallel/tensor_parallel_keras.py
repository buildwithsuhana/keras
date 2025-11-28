import logging
import gc
import os
import shutil
import tempfile
import ctypes
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

            # Ensure unique layer & model names so that when we assemble
            # multiple shard Models into a single Functional model, there
            # are no duplicate operation/layer names. Keras requires all
            # operation names to be unique within a graph.
            suffix = f"_tp{rank}"
            try:
                # Rename model
                shard._name = shard.name + suffix
            except Exception:
                pass
            # Rename layers
            for layer in getattr(shard, 'layers', []):
                try:
                    # Avoid doubling suffix if already applied
                    if not layer.name.endswith(suffix):
                        layer._name = layer.name + suffix
                except Exception:
                    # Be conservative: if a layer cannot be renamed, continue
                    continue

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
        # Note: We do NOT use assembled_model anymore.
        # Logic is handled in call()

    # --- SIMPLIFIED CALL METHOD ---
    def call(self, inputs, training=None, **kwargs):
        """
        Runs the forward pass on all shards and sums the results (Logits).
        This avoids graph name collisions because we don't build a static graph.
        """
        results = []
        for shard in self.model_shards:
            # Shards might return a dictionary or tensor. 
            # For CausalLM, it usually returns logits (tensor).
            out = shard(inputs, training=training, **kwargs)
            results.append(out)
        
        # Sum the outputs (Logits reduction for TP)
        total = results[0]
        for i in range(1, len(results)):
            total = ops.add(total, results[i])
            
        return total

    def _force_gc(self):
        gc.collect()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass

    def _build_shard_with_dummy_input(self, shard):
        dummy_inputs = {}
        for spec in self.input_specs:
            shape = list(spec["shape"])
            shape[0] = 1 
            shape = [s if s is not None else 1 for s in shape]
            clean_name = spec["name"].split(":")[0]
            dtype = spec["dtype"] or "float32"
            
            if "int" in str(dtype):
                data = ops.zeros(shape, dtype=dtype)
            else:
                data = ops.zeros(shape, dtype=dtype)
            dummy_inputs[clean_name] = data

        try:
            shard(dummy_inputs)
        except Exception:
            shard(list(dummy_inputs.values()))

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