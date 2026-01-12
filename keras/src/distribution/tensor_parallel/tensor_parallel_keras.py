import logging
import gc
import os
import shutil
import tempfile
import numpy as np
import psutil
import subprocess
import re
import keras
from keras import ops
from keras.src.distribution.tensor_parallel.autoconfig import get_default_config
from keras.src.distribution.tensor_parallel.coordinated_optimizer import TensorParallelOptimizer
from keras.src.distribution import list_devices
from keras.src.models import Model
import ctypes
import jax

# CRITICAL: Force low-memory policy for large models on Kaggle
keras.config.set_dtype_policy("bfloat16")

def log_mem_stats(rank, device_id, stage=""):
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 ** 2)
    print(f"📈 [Rank {rank}|{device_id}] {stage} | Host RSS: {mem_mb:.0f} MB")

class TensorParallelKeras(Model):
    def __init__(self, model, device_count=None, device_ids=None, **kwargs):
        super().__init__(**kwargs)

        def flush_memory():
            gc.collect()
            jax.clear_caches()
            try: ctypes.CDLL("libc.so.6").malloc_trim(0)
            except: pass

        def shred_variable(v):
            try: object.__setattr__(v, "_value", jax.numpy.zeros((0,), dtype=v.dtype))
            except: pass

        if device_count is None or device_ids is None:
            all_devices = list_devices()
            device_count = len(all_devices)
            device_ids = [str(d) for d in all_devices]

        self.device_count = device_count
        self.devices = [self._normalize_device_id(d) for d in device_ids]
        self.temp_dir = tempfile.mkdtemp(prefix="tp_weights_")
        
        # 1. Instantiate Master Model on CPU
        with jax.default_device(jax.devices("cpu")[0]):
            if callable(model) and not isinstance(model, keras.Model):
                loaded_model = model()
            else:
                loaded_model = model
            loaded_model.build({"token_ids": (None, 1), "padding_mask": (None, 1)})

        self.model_config = loaded_model.get_config()
        self.model_cls = loaded_model.__class__
        self.tensor_parallel_config = get_default_config(loaded_model, self.devices)

        # 2. Save and Shred ONE VARIABLE AT A TIME (Prevents peak RSS doubling)
        for v in loaded_model.variables:
            name = (v.path if hasattr(v, 'path') else v.name).replace("/", "_").replace(":", "_")
            path = os.path.join(self.temp_dir, name + ".npy")
            np.save(path, np.array(v)) # Use np.array to convert JAX -> NumPy
            shred_variable(v) # Free RAM immediately after saving
        
        del loaded_model 
        flush_memory()

        self.__dict__["model_shards"] = []
        from keras.src.distribution.tensor_parallel.parameter_sharding import make_parameter_sharded_model, ParameterShardingStrategy
        
        for rank, device_id in enumerate(self.devices):
            log_mem_stats(rank, device_id, "START creation")

            # Build shard on CPU
            with keras.device("cpu"):
                config = self.model_config.copy()
                config["name"] = f"shard_model_{rank}"
                shard = self.model_cls.from_config(config)
                shard.build({"token_ids": (None, 1), "padding_mask": (None, 1)})

            # Shard parameters
            shard, modified_vars = make_parameter_sharded_model(
                shard_model=shard,
                weight_loader=self._weight_loader, 
                config=self.tensor_parallel_config,
                rank=rank,
                device_count=self.device_count,
                device_id=device_id,
            )

            # Migration for Replicated weights
            strat_helper = ParameterShardingStrategy(self.device_count, rank)
            try:
                d_idx = int(str(device_id).split(":")[-1]) if ":" in str(device_id) else 0
                target_jax_device = jax.devices('gpu')[d_idx]
                var_to_owners = strat_helper._map_variables_to_owners(shard)
                
                for v in list(shard.variables):
                    v_name = v.path if hasattr(v, 'path') else v.name
                    # FIX: Correct naming match
                    clean_name = re.sub(r'^shard_model_\d+/', '', v_name).replace("/", ".")
                    if clean_name in modified_vars or v_name in modified_vars: 
                        continue 

                    lookup_name = re.sub(r'^shard_model_\d+/', '', v_name)
                    raw_val = self._weight_loader(lookup_name) 
                    if raw_val is not None:
                        val_gpu = jax.device_put(raw_val, target_jax_device)
                        if id(v) in var_to_owners:
                            for layer, attr_name, idx in var_to_owners[id(v)]:
                                strat_helper._replace_variable(layer, attr_name, v, val_gpu, index=idx)
                        else:
                            v.assign(val_gpu)
                        # NOTE: Don't shred assigned variables, they are now on GPU
            except Exception as e:
                print(f"⚠️ Migration Error rank {rank}: {e}")

            self.model_shards.append(shard)
            flush_memory() 
            log_mem_stats(rank, device_id, "DONE creation")

        try: shutil.rmtree(self.temp_dir)
        except: pass
        self.built = True

    # ... (rest of the Model class same as before)
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
        return {m.name: m.result() for m in self.metrics}, state

    def _weight_loader(self, param_name):
        name = param_name.replace("/", "_").replace(":", "_")
        path = os.path.join(self.temp_dir, name + ".npy")
        if not os.path.exists(path): return None
        val = np.load(path, mmap_mode='r')
        if hasattr(val, 'dtype') and ("V" in str(val.dtype) or "void" in str(val.dtype)):
            try:
                import ml_dtypes
                return val.view(ml_dtypes.bfloat16)
            except ImportError: return val.astype("float32")
        return val

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