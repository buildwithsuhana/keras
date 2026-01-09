import logging
import gc
import os
import tempfile
import shutil
import numpy as np
import keras
from keras import ops
from keras.src.distribution.tensor_parallel.autoconfig import get_default_config
from keras.src.distribution.tensor_parallel.parameter_sharding import make_parameter_sharded_model
from keras.src.distribution.tensor_parallel.coordinated_optimizer import TensorParallelOptimizer
from keras.src.distribution.tensor_parallel.parameter_sharding import (
    ParameterShardingStrategy,
)

class TensorParallelKeras(keras.Model):
    def __init__(self, model, device_count=None, device_ids=None, **kwargs):
        super().__init__(**kwargs)

        if device_count is None or device_ids is None:
            from keras.src.distribution import list_devices
            all_devices = list_devices()
            device_count = len(all_devices)
            device_ids = [str(d) for d in all_devices]

        self.device_count = device_count
        self.devices = [self._normalize_device_id(d) for d in device_ids]
        self.temp_dir = tempfile.mkdtemp(prefix="tp_weights_")
        
        # 1. Capture Config and Offload weights to CPU disk to save RAM
        if callable(model) and not isinstance(model, keras.Model):
            loaded_model = model()
        else:
            loaded_model = model

        self.model_config = loaded_model.get_config()
        self.model_cls = loaded_model.__class__
        self.tensor_parallel_config = get_default_config(loaded_model, self.devices)

        self._save_weights_to_disk(loaded_model)
        del loaded_model
        gc.collect()

        self.__dict__["model_shards"] = []

        # buildwithsuhana/keras/keras-ananta/keras/src/distribution/tensor_parallel/tensor_parallel_keras.py

        for rank, device_id in enumerate(self.devices):
            # FIX: Use a unique name scope to isolate variable identities in JAX
            with keras.name_scope(f"shard_{rank}"):
                with keras.device(device_id):
                    shard = self.model_cls.from_config(self.model_config)
                    if hasattr(shard, 'build_from_config'):
                         shard.build_from_config(self.model_config)

            # 1. Sharded Variables 
            shard, modified_vars = make_parameter_sharded_model(
                shard_model=shard,
                weight_loader=self._weight_loader, 
                config=self.tensor_parallel_config,
                rank=rank,
                device_count=self.device_count,
                device_id=device_id,
            )

            # 2. Unsharded Variables (Migration)
            strat_helper = ParameterShardingStrategy(self.device_count, rank)
            try:
                import jax
                target_device = jax.devices('gpu')[rank]
                var_to_owner = strat_helper._map_variables_to_owners(shard)
                
                for v in shard.variables:
                    v_name = v.path if hasattr(v, 'path') else v.name
                    if v_name in modified_vars: continue 

                    # FIX: Strip the 'shard_N/' name_scope prefix to find the weight on disk
                    lookup_name = v_name
                    if v_name.startswith(f"shard_{rank}/"):
                        lookup_name = v_name[len(f"shard_{rank}/"):]
                    
                    raw_val = self._weight_loader(lookup_name)
                    if raw_val is not None:
                        val_gpu = jax.device_put(raw_val, target_device)
                        if id(v) in var_to_owner:
                            layer, attr_name = var_to_owner[id(v)]
                            strat_helper._replace_variable(layer, attr_name, v, val_gpu, device_id=device_id)
            except Exception as e:
                print(f"⚠️ Migration Error: {e}")

            self.model_shards.append(shard)

        try: shutil.rmtree(self.temp_dir)
        except: pass
        self.built = True

    def _normalize_device_id(self, device_id):
        d_str = str(device_id).lower()
        if "gpu" in d_str:
            return f"gpu:{d_str.split(':')[-1]}" if ":" in d_str else f"gpu:{d_str}"
        return d_str

    def call(self, inputs, training=None, **kwargs):
        results = []
        for i, shard in enumerate(self.model_shards):
            with keras.device(self.devices[i]):
                out = shard(inputs, training=training, **kwargs)
                results.append(out)
        
        # Parallel reduction (Add) of shard outputs
        total = results[0]
        for i in range(1, len(results)):
            total = ops.add(total, results[i])
        return total

    def _save_weights_to_disk(self, model):
        for v in model.variables:
            name = v.path if hasattr(v, 'path') else v.name
            path = os.path.join(self.temp_dir, name.replace("/", "_").replace(":", "_") + ".npy")
            val = v.numpy()
            if hasattr(val, 'dtype') and val.dtype.name == 'bfloat16':
                val = val.astype('float32')
            np.save(path, val)

    def _weight_loader(self, param_name):
        path = os.path.join(self.temp_dir, param_name.replace("/", "_").replace(":", "_") + ".npy")
        if os.path.exists(path): return np.load(path, mmap_mode='r')
        return None