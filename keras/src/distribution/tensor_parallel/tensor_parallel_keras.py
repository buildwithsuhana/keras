import logging
import gc
import os
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
        
        # --- 1. MEMORY PRESERVATION ---
        self.temp_dir = tempfile.mkdtemp(prefix="tp_weights_")
        
        if callable(model) and not isinstance(model, keras.Model):
            print("🏭 Executing Model Factory to load Master Model...")
            loaded_model = model()
        else:
            loaded_model = model

        self.model_config = loaded_model.get_config()
        self.model_cls = loaded_model.__class__
        
        # Capture Inputs for manual building
        # We store the batch shape and dtype to create dummy inputs later
        self.input_specs = []
        if hasattr(loaded_model, "inputs") and loaded_model.inputs:
             for i in loaded_model.inputs:
                 self.input_specs.append({"shape": i.shape, "dtype": i.dtype, "name": i.name})

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
                # FIX: Modify config name to ensure uniqueness in the graph
                shard_config = self.model_config.copy()
                original_name = shard_config.get("name", "model")
                shard_config["name"] = f"shard_{rank}_{original_name}"
                
                try:
                    shard = self.model_cls.from_config(shard_config)
                except Exception:
                    # Fallback for models that might reject modified config
                    shard = self.model_cls.from_config(self.model_config)
                
                # Force uniqueness (Double check)
                shard._name = f"shard_{rank}_{original_name}"
                
                # FIX: Force Build to initialize variables
                # KerasNLP models often need a real call or build to create weights
                if not shard.built and self.input_specs:
                    try:
                        self._build_shard_with_dummy_input(shard)
                    except Exception as e:
                        print(f"⚠️ Manual build warning: {e}. Trying auto-build...")

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

    def _build_shard_with_dummy_input(self, shard):
        """Creates dummy inputs to force the model to build variables."""
        dummy_inputs = {}
        for spec in self.input_specs:
            # Create a batch of 1 with the correct shape/dtype
            shape = list(spec["shape"])
            shape[0] = 1 # Batch size 1
            # Handle None dimensions
            shape = [s if s is not None else 1 for s in shape]
            
            clean_name = spec["name"].split(":")[0]
            
            # Special handling for token_ids/padding_mask which are usually int32
            dtype = spec["dtype"] or "float32"
            if "int" in dtype:
                data = ops.zeros(shape, dtype=dtype)
            else:
                data = ops.zeros(shape, dtype=dtype)
                
            dummy_inputs[clean_name] = data

        # Call the model to initialize weights
        # KerasNLP models often expect dict inputs
        try:
            shard(dummy_inputs)
        except Exception:
            # Fallback to positional list
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

    def build_assembled_model(self):
        ref = self.model_shards[0]
        
        # Reconstruct inputs
        inputs = {}
        # Use captured specs if available, else inspect reference
        specs = self.input_specs if self.input_specs else []
        
        if not specs and hasattr(ref, "inputs"):
             for i in ref.inputs:
                 specs.append({"shape": i.shape, "dtype": i.dtype, "name": i.name})

        # Create symbolic inputs for the functional graph
        symbolic_inputs = {}
        for spec in specs:
            name = spec["name"].split(':')[0]
            symbolic_inputs[name] = keras.Input(
                shape=spec["shape"][1:], 
                dtype=spec["dtype"], 
                name=name
            )

        shard_outputs = []
        for shard in self.model_shards:
            # Call shard with symbolic inputs
            try:
                # Try dict input first
                shard_out = shard(symbolic_inputs)
            except Exception:
                # Fallback to list input
                shard_out = shard(list(symbolic_inputs.values()))
            
            shard_outputs.append(shard_out)
        
        # Reduce outputs
        if len(shard_outputs) > 1:
            out = keras.layers.Add()(shard_outputs)
        else:
            out = shard_outputs[0]
            
        return keras.Model(inputs=symbolic_inputs, outputs=out)

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