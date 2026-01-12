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
from keras.src.distribution.tensor_parallel.coordinated_optimizer import TensorParallelOptimizer
from keras.src.distribution import list_devices
from keras.src.models import Model
import ctypes
import re
import jax
from collections import namedtuple

# Virtual object to satisfy CoordinatedOptimizer expectations without full Models
VirtualShard = namedtuple('VirtualShard', ['optimizer', 'trainable_variables'])

def log_mem_stats(rank, device_id, stage=""):
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 ** 2)
    print(f"📈 [Rank {rank}|{device_id}] {stage} | Host RSS: {mem_mb:.0f} MB")

class TensorParallelKeras(Model):
    def __init__(self, model, device_count=None, device_ids=None, **kwargs):
        super().__init__(**kwargs)

        def flush_memory():
            gc.collect()
            jax.clear_caches() # Critical: Frees Host RAM used for GPU transfers
            try: ctypes.CDLL("libc.so.6").malloc_trim(0)
            except: pass

        def shred_variable(v):
            """Bypasses shape checks to release physical RAM allocation."""
            try:
                object.__setattr__(v, "_value", jax.numpy.zeros((0,), dtype=v.dtype))
            except: pass

        if device_count is None or device_ids is None:
            all_devices = list_devices()
            device_count = len(all_devices)
            device_ids = [str(d) for d in all_devices]

        self.device_count = device_count
        self.devices = [self._normalize_device_id(d) for d in device_ids]
        self.temp_dir = tempfile.mkdtemp(prefix="tp_weights_")
        
        # 1. Build the structure-only model on CPU
        with jax.default_device(jax.devices("cpu")[0]):
            if callable(model) and not isinstance(model, keras.Model):
                loaded_model = model()
            else:
                loaded_model = model
            loaded_model.build({"token_ids": (None, 1), "padding_mask": (None, 1)})

        self.model_config = loaded_model.get_config()
        self.model_cls = loaded_model.__class__
        self.tensor_parallel_config = get_default_config(loaded_model, self.devices)

        # 2. Sequential Sharding - Stateless Parameter Creation
        self._save_weights_to_disk(loaded_model)
        
        # Free 18GB Host RAM immediately by orphaning CPU buffers
        for v in loaded_model.variables:
            shred_variable(v)
        
        self.__dict__["structure_model"] = loaded_model 
        self.__dict__["shard_params"] = [] 
        self.__dict__["shard_optimizers"] = []
        flush_memory()

        # 3. Create GPU-resident Variable sets sequentially
        for rank, device_id in enumerate(self.devices):
            log_mem_stats(rank, device_id, f"START Parameter Creation rank {rank}")
            
            d_idx = int(str(device_id).split(":")[-1]) if ":" in str(device_id) else 0
            target_jax_device = jax.devices('gpu')[d_idx]

            shard_trainable = []
            shard_non_trainable = []

            for v in self.structure_model.variables:
                v_name = v.path if hasattr(v, 'path') else v.name
                
                action = None
                for pattern, act in self.tensor_parallel_config.state_rules.items():
                    if re.search(pattern, v_name):
                        action = act
                        break
                
                raw_val = self._weight_loader(v_name)
                if raw_val is not None:
                    processed_val = action(raw_val, rank) if (action and callable(action)) else raw_val
                    
                    with keras.device(device_id):
                        new_v = keras.Variable(
                            initializer=jax.device_put(processed_val, target_jax_device),
                            dtype=v.dtype,
                            trainable=v.trainable,
                            name=f"rank_{rank}_{v_name}"
                        )
                    
                    if v.trainable: shard_trainable.append(new_v)
                    else: shard_non_trainable.append(new_v)
                    del raw_val, processed_val
            
            self.shard_params.append((shard_trainable, shard_non_trainable))
            flush_memory() 
            log_mem_stats(rank, device_id, f"DONE Parameter Creation rank {rank}")

        try: shutil.rmtree(self.temp_dir)
        except: pass
        self.built = True

    def _normalize_device_id(self, device_id):
        d_str = str(device_id).lower()
        if "tpu" in d_str: return f"tpu:{d_str.split(':')[-1]}" if ":" in d_str else f"tpu:{d_str}"
        return f"gpu:{d_str.split(':')[-1]}" if ":" in d_str else f"gpu:{d_str}"

    def call(self, inputs, training=None, **kwargs):
        results = []
        for i, (trainable, non_trainable) in enumerate(self.shard_params):
            with keras.device(self.devices[i]):
                y_pred, _ = self.structure_model.stateless_call(
                    trainable, non_trainable, inputs, training=training, **kwargs
                )
                results.append(y_pred)
        
        total = results[0]
        for i in range(1, len(results)): total = ops.add(total, results[i])
        return total
    
    def train_step(self, state, data):
        """Fixed signature for Keras 3 JAX backend."""
        from keras.src.trainers.data_adapters import data_adapter_utils
        x, y, sample_weight = data_adapter_utils.unpack_x_y_sample_weight(data)
        all_shard_grads_vars = []
        total_loss = None

        for i, (trainable, non_trainable) in enumerate(self.shard_params):
            with keras.device(self.devices[i]):
                t_vals = [v.value for v in trainable]
                nt_vals = [v.value for v in non_trainable]
                
                def compute_loss(tv, ntv, xd, yd):
                    yp, ntu = self.structure_model.stateless_call(tv, ntv, xd, training=True)
                    l = self.compute_loss(x=xd, y=yd, y_pred=yp, sample_weight=sample_weight)
                    return l, ntu
                
                (loss_val, _), grads = jax.value_and_grad(compute_loss, has_aux=True)(t_vals, nt_vals, x, y)
                all_shard_grads_vars.append(list(zip(grads, trainable)))
                if total_loss is None: total_loss = loss_val

        self.optimizer.apply_gradients(all_shard_grads_vars, shard_models=self.shard_optimizers)
        
        for metric in self.metrics:
            if metric.name == "loss": metric.update_state(total_loss)
        return {m.name: m.result() for m in self.metrics}

    def compile(self, optimizer=None, **kwargs):
        if optimizer:
            self.shard_optimizers.clear()
            for i in range(self.device_count):
                shard_opt = keras.optimizers.get(optimizer) if isinstance(optimizer, str) else optimizer.from_config(optimizer.get_config())
                self.shard_optimizers.append(VirtualShard(optimizer=shard_opt, trainable_variables=self.shard_params[i][0]))

            opt = TensorParallelOptimizer(optimizer, self.device_count)
            var_map = {}
            for i, (trainable, _) in enumerate(self.shard_params):
                for v in trainable:
                    key = v.name.split("_", 2)[-1] 
                    if key not in var_map: var_map[key] = [None] * self.device_count
                    var_map[key][i] = v
            opt.__dict__["_shard_var_map"] = var_map
            super().compile(optimizer=opt, **kwargs)
        else: super().compile(optimizer=optimizer, **kwargs)

    def _save_weights_to_disk(self, model):
        for v in model.variables:
            name = (v.path if hasattr(v, 'path') else v.name).replace("/", "_").replace(":", "_")
            path = os.path.join(self.temp_dir, name + ".npy")
            np.save(path, v.numpy())

    def _weight_loader(self, param_name):
        name = param_name.replace("/", "_").replace(":", "_")
        path = os.path.join(self.temp_dir, name + ".npy")
        return np.load(path, mmap_mode='r') if os.path.exists(path) else None