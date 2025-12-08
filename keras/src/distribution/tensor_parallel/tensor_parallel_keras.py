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
from keras.src.trainers.data_adapters import data_adapter_utils
import ctypes

logger = logging.getLogger(__file__)

class TensorParallelKeras(Model):
    def __init__(self, model, device_count=None, device_ids=None, **kwargs):
        super().__init__(**kwargs)

        def flush_memory():
            gc.collect()
            try: ctypes.CDLL("libc.so.6").malloc_trim(0)
            except: pass

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
        flush_memory()

        self.model_shards = []
        print(f"🚀 Initializing TP on {self.devices}")

        # Helper for migration
        from keras.src.distribution.tensor_parallel.parameter_sharding import ParameterShardingStrategy
        strat_helper = ParameterShardingStrategy(1, 0)

        for rank, device_id in enumerate(self.devices):
            print(f"[{device_id}] ⏳ Creating shard {rank+1}/{self.device_count}...")
            
            with keras.device("cpu"):
                shard = self.model_cls.from_config(self.model_config)
                if hasattr(shard, 'build_from_config'):
                     shard.build_from_config(self.model_config)

            shard, modified_vars = make_parameter_sharded_model(
                shard_model=shard,
                weight_loader=self._weight_loader, 
                config=self.tensor_parallel_config,
                rank=rank,
                device_count=self.device_count,
                device_id=device_id,
            )

            # Migration of unsharded variables
            try:
                import jax
                d_str = str(device_id)
                idx = int(d_str.split(":")[-1]) if ":" in d_str else 0
                try: target_device = jax.devices('gpu')[idx]
                except: target_device = jax.devices()[idx]
                
                var_to_owner = strat_helper._map_variables_to_owners(shard)
                
                migrated_count = 0
                for v in shard.variables:
                    v_name = v.path if hasattr(v, 'path') else v.name
                    if v_name in modified_vars: continue 

                    val_gpu = jax.device_put(v.value, target_device)
                    
                    if id(v) in var_to_owner:
                        layer, attr_name = var_to_owner[id(v)]
                        strat_helper._replace_variable(layer, attr_name, v, val_gpu)
                        migrated_count += 1
            except Exception as e:
                print(f"⚠️ Migration Error: {e}")

            self.model_shards.append(shard)
            flush_memory()
            print(f"[{device_id}] ✅ Shard ready.")

        try: shutil.rmtree(self.temp_dir)
        except: pass
        self.built = True

    def _normalize_device_id(self, device_id):
        d_str = str(device_id).lower()
        if "tpu" in d_str and "gpu" not in d_str:
            if d_str.startswith("tpu:"): return d_str
            match = re.search(r"tpu_(\d+)", d_str)
            if match: return f"tpu:{match.group(1)}"
            if ":" not in d_str: return f"tpu:{d_str}"
            return d_str
        if ":" not in d_str: return f"gpu:{d_str}"
        return d_str

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
    
    def train_step(self, state, data):
        # In JAX backend, train_step receives (state, data).
        # We ignore 'state' because our real variables are distributed in self.model_shards
        # and we are running in eager mode (side-effects allowed).
        
        x, y, sample_weight = data_adapter_utils.unpack_x_y_sample_weight(data)
        
        all_shard_grads_vars = []
        total_loss = 0.0
        import tensorflow as tf
        
        # 1. Forward & Backward on Shards
        for i, shard in enumerate(self.model_shards):
            with keras.device(self.devices[i]):
                with tf.GradientTape() as tape:
                    # Forward pass
                    y_pred = shard(x, training=True)
                    
                    # Compute loss using the correct method
                    # Option A: Use compute_loss (Recommended - handles all losses)
                    loss = self.compute_loss(x, y, y_pred, sample_weight)
                    
                    # Option B: Use _compile_loss directly (Only uses the loss function passed to compile)
                    # loss = self._compile_loss(y, y_pred, sample_weight)

                # Compute gradients
                trainable_vars = shard.trainable_variables
                grads = tape.gradient(loss, trainable_vars)
                
                # Store (grad, var) pairs for this shard
                all_shard_grads_vars.append(list(zip(grads, trainable_vars)))
                
                if i == 0:
                    total_loss = loss

        # 2. Apply Gradients (via Coordinator -> Shards)
        self.optimizer.apply_gradients(all_shard_grads_vars, shard_models=self.model_shards)
        
        # 3. Update Metrics
        # Manually update loss metric
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(total_loss)
            # Add other metrics here if needed (requires synchronization)
        
        logs = {m.name: m.result() for m in self.metrics}
        
        # Return logs and the UNCHANGED state (since the coordinator has no state)
        return logs, state

    def _save_weights_to_disk(self, model):
        for v in model.variables:
            name = v.path if hasattr(v, 'path') else v.name
            safe_name = name.replace("/", "_").replace(":", "_")
            path = os.path.join(self.temp_dir, safe_name + ".npy")
            val = v.numpy()
            if (hasattr(val, 'dtype') and (val.dtype.name == 'bfloat16' or str(val.dtype) == 'bfloat16')) or \
               (val.dtype.char == 'V' and val.itemsize == 2):
                val = val.astype('float32')
            np.save(path, val)

    def _weight_loader(self, param_name):
        safe_name = param_name.replace("/", "_").replace(":", "_")
        path = os.path.join(self.temp_dir, safe_name + ".npy")
        if os.path.exists(path): return np.load(path, mmap_mode='r')
        return None

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