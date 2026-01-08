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
        
        # 1. Capture Config and Offload
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

        for rank, device_id in enumerate(self.devices):
            print(f"[{device_id}] ⏳ Anchoring shard {rank+1}/{self.device_count} to hardware...")
            
            with keras.device(device_id):
                # FIXED: Force variable instantiation ON GPU immediately
                shard = self.model_cls.from_config(self.model_config)
                
                # We trigger a dummy call with symbolic tensors to force 
                # the backend to build the model and variables on the GPU.
                if hasattr(shard, 'build_from_config'):
                    shard.build_from_config(self.model_config)
                
                # 2. Shard/Replicate parameters
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