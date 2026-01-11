# buildwithsuhana/keras/keras-ananta/keras/src/distribution/tensor_parallel/tensor_parallel_keras.py

import logging
import gc
import os
import shutil
import tempfile
import numpy as np
import psutil
import subprocess
import keras
from keras import ops
from keras.src.distribution.tensor_parallel.autoconfig import get_default_config
from keras.src.distribution.tensor_parallel.parameter_sharding import make_parameter_sharded_model, ParameterShardingStrategy
from keras.src.distribution.tensor_parallel.coordinated_optimizer import TensorParallelOptimizer
from keras.src.distribution import list_devices
from keras.src.models import Model
import ctypes

def log_mem_stats(rank, device_id, stage=""):
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 ** 2)
    print(f"📈 [Shard {rank}|{device_id}] {stage} | Host RSS: {mem_mb:.0f} MB")

class TensorParallelKeras(Model):
    def __init__(self, model, device_count=None, device_ids=None, **kwargs):
        super().__init__(**kwargs)

        def flush_memory():
            import jax
            gc.collect()
            # CRITICAL: Clears JAX's internal buffer caches that hold Host RAM
            jax.clear_caches() 
            try: ctypes.CDLL("libc.so.6").malloc_trim(0)
            except: pass

        if device_count is None or device_ids is None:
            all_devices = list_devices()
            device_count = len(all_devices)
            device_ids = [str(d) for d in all_devices]

        self.device_count = device_count
        self.devices = [self._normalize_device_id(d) for d in device_ids]
        self.temp_dir = tempfile.mkdtemp(prefix="tp_weights_")
        
        # 1. Master Model Cleanup
        if callable(model) and not isinstance(model, keras.Model):
            loaded_model = model()
        else:
            loaded_model = model

        self.model_config = loaded_model.get_config()
        self.model_cls = loaded_model.__class__
        self.tensor_parallel_config = get_default_config(loaded_model, self.devices)

        self._save_weights_to_disk(loaded_model)
        del loaded_model 
        flush_memory()

        self.__dict__["model_shards"] = []

        # 2. Sequential Shard Creation
        # ... (around line 52)
        # 2. Sequential Shard Creation
        for rank, device_id in enumerate(self.devices):
            log_mem_stats(rank, device_id, "START creation")

            with keras.name_scope(f"shard_{rank}"):
                # Now this will use 0MB RAM/VRAM
                with keras.device("meta"):
                    config = self.model_config.copy()
                    config["name"] = f"shard_model_{rank}"
                    shard = self.model_cls.from_config(config)
                    shard.build({"token_ids": (None, 1), "padding_mask": (None, 1)})

            # 3. Sharding (Destructive)
            # This utility moves slices from disk -> GPU and replaces 'meta' vars
            shard, modified_vars = make_parameter_sharded_model(
                shard_model=shard,
                weight_loader=self._weight_loader, 
                config=self.tensor_parallel_config,
                rank=rank,
                device_count=self.device_count,
                device_id=device_id,
            )

            # 4. Migration for Replicated weights (Destructive)
            from keras.src.distribution.tensor_parallel.parameter_sharding import ParameterShardingStrategy
            strat_helper = ParameterShardingStrategy(self.device_count, rank)
            try:
                import jax
                d_idx = int(str(device_id).split(":")[-1]) if ":" in str(device_id) else 0
                target_jax_device = jax.devices('gpu')[d_idx]
                var_to_owners = strat_helper._map_variables_to_owners(shard)
                
                # --- START OF YOUR SNIPPET ---
                vars_to_process = list(shard.variables)
                for v in vars_to_process:
                    v_name = v.path if hasattr(v, 'path') else v.name
                    if v_name in modified_vars: continue 

                    lookup_name = v_name.replace(f"shard_{rank}/", "")
                    raw_val = self._weight_loader(lookup_name)
                    if raw_val is not None:
                        # Move the replicated weight directly to the specific GPU
                        val_gpu = jax.device_put(raw_val, target_jax_device)
                        
                        if id(v) in var_to_owners:
                            for layer, attr_name in var_to_owners[id(v)]:
                                # Use helper to replace 'meta' Variable with the GPU one
                                strat_helper._replace_variable(
                                    layer, attr_name, v, val_gpu, device_id=device_id
                                )
                        else:
                            v.assign(val_gpu)
                        
                        del raw_val, val_gpu
                        gc.collect()
                # --- END OF YOUR SNIPPET ---

            except Exception as e:
                print(f"⚠️ Migration Error rank {rank}: {e}")

            self.model_shards.append(shard)
            flush_memory() 
            log_mem_stats(rank, device_id, "DONE creation")


        try: shutil.rmtree(self.temp_dir)
        except: pass
        self.built = True

    def _normalize_device_id(self, device_id):
        d_str = str(device_id).lower()
        if "tpu" in d_str: return f"tpu:{d_str.split(':')[-1]}" if ":" in d_str else f"tpu:{d_str}"
        return f"gpu:{d_str.split(':')[-1]}" if ":" in d_str else f"gpu:{d_str}"

    def call(self, inputs, training=None, **kwargs):
        results = [shard(inputs, training=training, **kwargs) for shard in self.model_shards]
        total = results[0]
        for i in range(1, len(results)): total = ops.add(total, results[i])
        return total
    
    def train_step(self, state, data):
        import jax
        from keras.src.trainers.data_adapters import data_adapter_utils
        x, y, sample_weight = data_adapter_utils.unpack_x_y_sample_weight(data)
        all_shard_grads_vars = []
        total_loss = 0.0

        for i, shard in enumerate(self.model_shards):
            with keras.device(self.devices[i]):
                t_vars = [v.value for v in shard.trainable_variables]
                nt_vars = [v.value for v in shard.non_trainable_variables]
                def compute_loss(tv, ntv, xd, yd):
                    yp, ntu = shard.stateless_call(tv, ntv, xd, training=True)
                    l = self.compute_loss(x=xd, y=yd, y_pred=yp, sample_weight=sample_weight)
                    return l, ntu
                (loss_val, _), grads = jax.value_and_grad(compute_loss, has_aux=True)(t_vars, nt_vars, x, y)
                all_shard_grads_vars.append(list(zip(grads, shard.trainable_variables)))
                if i == 0: total_loss = loss_val

        self.optimizer.apply_gradients(all_shard_grads_vars, shard_models=self.model_shards)
        for metric in self.metrics:
            if metric.name == "loss": metric.update_state(total_loss)
        return {m.name: m.result() for m in self.metrics}

    def _save_weights_to_disk(self, model):
        """Saves weights without transient float32 upcast."""
        for v in model.variables:
            name = (v.path if hasattr(v, 'path') else v.name).replace("/", "_").replace(":", "_")
            path = os.path.join(self.temp_dir, name + ".npy")
            val = v.numpy() # Stays bf16 if Keras is in bf16 mode
            np.save(path, val)
            del val

    def _weight_loader(self, param_name):
        name = param_name.replace("/", "_").replace(":", "_")
        path = os.path.join(self.temp_dir, name + ".npy")
        return np.load(path, mmap_mode='r') if os.path.exists(path) else None

    def compile(self, optimizer=None, **kwargs):
        if len(self.model_shards) > 1 and optimizer:
            opt = TensorParallelOptimizer(optimizer, self.device_count)
            opt.__dict__["_shard_models"] = self.model_shards
            var_map = {}
            for i, shard in enumerate(self.model_shards):
                for v in shard.trainable_variables:
                    key = v.path if hasattr(v, "path") else v.name
                    if key not in var_map: var_map[key] = [None]*self.device_count
                    var_map[key][i] = v
            opt.__dict__["_shard_var_map"] = var_map
            super().compile(optimizer=opt, **kwargs)
            for i, shard in enumerate(self.model_shards):
                shard_opt = keras.optimizers.get(optimizer) if isinstance(optimizer, str) else optimizer.from_config(optimizer.get_config())
                with keras.device(self.devices[i]): shard.compile(optimizer=shard_opt, **kwargs)
        else: super().compile(optimizer=optimizer, **kwargs)