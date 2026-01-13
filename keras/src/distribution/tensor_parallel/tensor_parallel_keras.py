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
import ctypes
import jax
from keras.src.distribution.tensor_parallel.autoconfig import get_default_config
from keras.src.distribution.tensor_parallel.coordinated_optimizer import TensorParallelOptimizer
from keras.src.models import Model

# Force bfloat16 for high-memory efficiency
keras.config.set_dtype_policy("bfloat16")

class TensorParallelKeras(Model):
    def __init__(self, model, device_count=None, device_ids=None, **kwargs):
        super().__init__(**kwargs)

        def flush_memory():
            gc.collect()
            jax.clear_caches()
            try: ctypes.CDLL("libc.so.6").malloc_trim(0)
            except: pass

        if device_count is None or device_ids is None:
            from keras.src.distribution import list_devices
            all_devices = list_devices()
            device_count = len(all_devices)
            device_ids = [str(d) for d in all_devices]

        self.device_count = device_count
        self.devices = [str(d) for d in device_ids]
        self.temp_dir = tempfile.mkdtemp(prefix="tp_weights_")
        
        # 1. Build Master Model on CPU
        with jax.default_device(jax.devices("cpu")[0]):
            loaded_model = model() if callable(model) else model
            loaded_model.build({"token_ids": (None, 1), "padding_mask": (None, 1)})

        self.model_config = loaded_model.get_config()
        self.model_cls = loaded_model.__class__
        self.tensor_parallel_config = get_default_config(loaded_model, self.devices)

        # 2. Save variables to disk and shred CPU memory
        for v in loaded_model.variables:
            name = (v.path if hasattr(v, 'path') else v.name).replace("/", "_").replace(":", "_")
            np.save(os.path.join(self.temp_dir, name + ".npy"), np.array(v))
            try: object.__setattr__(v, "_value", jax.numpy.zeros((0,), dtype=v.dtype))
            except: pass
        
        del loaded_model 
        flush_memory()

        # 3. Create Model Shards
        self.__dict__["model_shards"] = []
        from keras.src.distribution.tensor_parallel.parameter_sharding import make_parameter_sharded_model, ParameterShardingStrategy
        
        for rank, device_id in enumerate(self.devices):
            with keras.device("cpu"):
                config = self.model_config.copy()
                config["name"] = f"shard_model_{rank}"
                shard = self.model_cls.from_config(config)
                shard.build({"token_ids": (None, 1), "padding_mask": (None, 1)})

            shard, modified_vars = make_parameter_sharded_model(
                shard_model=shard, weight_loader=self._weight_loader, 
                config=self.tensor_parallel_config, rank=rank,
                device_count=self.device_count, device_id=device_id,
            )

            # Replicate weights not covered by sharding rules
            strat_helper = ParameterShardingStrategy(self.device_count, rank)
            d_idx = int(str(device_id).split(":")[-1]) if ":" in str(device_id) else 0
            target_jax_device = jax.devices('gpu')[d_idx]
            var_to_owners = strat_helper._map_variables_to_owners(shard)
            
            for v in list(shard.variables):
                v_name = v.path if hasattr(v, 'path') else v.name
                if v_name in modified_vars: continue 

                lookup_name = re.sub(r'^shard_model_\d+/', '', v_name)
                raw_val = self._weight_loader(lookup_name) 
                if raw_val is not None:
                    val_gpu = jax.device_put(raw_val, target_jax_device)
                    if id(v) in var_to_owners:
                        for layer, attr_name, idx in var_to_owners[id(v)]:
                            strat_helper._replace_variable(layer, attr_name, v, val_gpu, index=idx)
                    
                    # Memory Logging for Replication
                    mem_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
                    print(f"🔄 [Rank {rank}] Replicated layer variable: {v_name} | Host RSS: {mem_mb:.0f} MB")

            self.model_shards.append(shard)
            flush_memory() 

        try: shutil.rmtree(self.temp_dir)
        except: pass
        self.built = True

    def _weight_loader(self, param_name):
        name = param_name.replace("/", "_").replace(":", "_")
        path = os.path.join(self.temp_dir, name + ".npy")
        if not os.path.exists(path): return None
        return np.load(path)

    def call(self, inputs, training=None, **kwargs):
        results = [shard(inputs, training=training, **kwargs) for shard in self.model_shards]
        total = results[0]
        for i in range(1, len(results)): total = ops.add(total, results[i])
        return total

    def compile(self, optimizer=None, **kwargs):
        opt = TensorParallelOptimizer(optimizer, self.device_count)
        opt.__dict__["_shard_models"] = self.model_shards
        super().compile(optimizer=opt, **kwargs)