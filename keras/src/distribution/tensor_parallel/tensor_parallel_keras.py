import logging
import gc
import os
import re
import shutil
import tempfile
import numpy as np
import psutil
import subprocess
import keras
from keras import ops
from keras.src.distribution.tensor_parallel.autoconfig import get_default_config
from keras.src.distribution.tensor_parallel.parameter_sharding import make_parameter_sharded_model
from keras.src.distribution.tensor_parallel.coordinated_optimizer import TensorParallelOptimizer
from keras.src.distribution import list_devices
from keras.src.models import Model
import ctypes

logger = logging.getLogger(__file__)

def log_mem_stats(rank, device_id, stage=""):
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 ** 2)
    gpu_str = ""
    try:
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'], encoding='utf-8')
        mems = [int(x) for x in result.strip().split('\n') if x.strip()]
        for i, m in enumerate(mems): gpu_str += f"G{i}:{m}MB "
    except: gpu_str = "N/A"
    print(f"📈 [Shard {rank}|{device_id}] {stage} | Host RSS: {mem_mb:.0f} MB | {gpu_str}")

class TensorParallelKeras(Model):
    def __init__(self, model, device_count=None, device_ids=None, **kwargs):
        super().__init__(**kwargs)

        def flush_memory():
            gc.collect()
            try:
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except:
                pass

        if device_count is None or device_ids is None:
            all_devices = list_devices()
            device_count = len(all_devices)
            device_ids = [str(d) for d in all_devices]

        self.device_count = device_count
        self.devices = [self._normalize_device_id(d) for d in device_ids]
        self.temp_dir = tempfile.mkdtemp(prefix="tp_weights_")
        
        # 1. Instantiate or reference the master model
        if callable(model) and not isinstance(model, keras.Model):
            print("🏭 Executing Model Factory...")
            loaded_model = model()
        else:
            loaded_model = model

        self.model_config = loaded_model.get_config()
        self.model_cls = loaded_model.__class__
        self.tensor_parallel_config = get_default_config(loaded_model, self.devices)

        # 2. Offload to disk and PURGE CPU copy immediately
        print(f"💾 Offloading {len(loaded_model.variables)} variables to disk...")
        self._save_weights_to_disk(loaded_model)
        
        print("🗑️ Destroying Master Model to free CPU RAM...")
        del loaded_model
        flush_memory()

        self.__dict__["model_shards"] = []
        print(f"🚀 Initializing TP on {self.devices}")

        from keras.src.distribution.tensor_parallel.parameter_sharding import ParameterShardingStrategy
        
        for rank, device_id in enumerate(self.devices):
            log_mem_stats(rank, device_id, "START creation")

            # Create the blueprint of the model on CPU
            with keras.name_scope(f"shard_{rank}"):
                with keras.device("cpu"):
                    config = self.model_config.copy()
                    config["name"] = f"shard_model_{rank}"
                    shard = self.model_cls.from_config(config)
                    # Use a minimal build to avoid temporary memory spikes
                    shard.build({"token_ids": (None, 1), "padding_mask": (None, 1)})

            # Sharding logic splits the variables
            shard, modified_vars = make_parameter_sharded_model(
                shard_model=shard,
                weight_loader=self._weight_loader, 
                config=self.tensor_parallel_config,
                rank=rank,
                device_count=self.device_count,
                device_id=device_id,
            )

            # 4. Migration: Move CPU variables to GPU and release CPU buffers
            strat_helper = ParameterShardingStrategy(self.device_count, rank)
            try:
                import jax
                target_jax_device = jax.devices('gpu')[rank]
                var_to_owners = strat_helper._map_variables_to_owners(shard)
                
                for v in shard.variables:
                    v_name = v.path if hasattr(v, 'path') else v.name
                    # Skip if already handled by parameter_sharding (e.g. sharded weights)
                    if v_name in modified_vars: continue 

                    lookup_name = v_name
                    if v_name.startswith(f"shard_{rank}/"):
                        lookup_name = v_name[len(f"shard_{rank}/"):]

                    raw_val = self._weight_loader(lookup_name)
                    if raw_val is not None:
                        # CRITICAL: device_put moves to GPU; v.assign replaces CPU buffer
                        val_gpu = jax.device_put(raw_val, target_jax_device)
                        v.assign(val_gpu)
                        
                        # Handle duplicate references (owners)
                        if id(v) in var_to_owners:
                            for layer, attr_name in var_to_owners[id(v)]:
                                strat_helper._replace_variable(layer, attr_name, v, val_gpu, device_id=device_id)
                        del raw_val # release file handle/buffer
            except Exception as e:
                print(f"⚠️ Migration Error Shard {rank}: {e}")

            self.model_shards.append(shard)
            flush_memory() # Clear the specific CPU blueprint just sharded
            log_mem_stats(rank, device_id, "DONE creation")

        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass
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
        import jax
        from keras.src.trainers.data_adapters import data_adapter_utils

        x, y, sample_weight = data_adapter_utils.unpack_x_y_sample_weight(data)
        all_shard_grads_vars = []
        total_loss = 0.0

        for i, shard in enumerate(self.model_shards):
            with keras.device(self.devices[i]):
                trainable_values = [v.value for v in shard.trainable_variables]
                non_trainable_values = [v.value for v in shard.non_trainable_variables]

                def compute_loss(t_vars, nt_vars, x_data, y_data):
                    y_pred, nt_updates = shard.stateless_call(t_vars, nt_vars, x_data, training=True)
                    loss = self.compute_loss(x=x_data, y=y_data, y_pred=y_pred, sample_weight=sample_weight)
                    return loss, nt_updates

                (loss_val, _), grads = jax.value_and_grad(compute_loss, has_aux=True)(
                    trainable_values, non_trainable_values, x, y
                )
                
                all_shard_grads_vars.append(list(zip(grads, shard.trainable_variables)))
                if i == 0: total_loss = loss_val

        self.optimizer.apply_gradients(all_shard_grads_vars, shard_models=self.model_shards)
        for metric in self.metrics:
            if metric.name == "loss": metric.update_state(total_loss)
        
        return {m.name: m.result() for m in self.metrics}

    def _save_weights_to_disk(self, model):
        """Saves weights without doubling RAM via float32 conversion."""
        for v in model.variables:
            name = v.path if hasattr(v, 'path') else v.name
            safe_name = name.replace("/", "_").replace(":", "_")
            path = os.path.join(self.temp_dir, safe_name + ".npy")
            
            val = v.numpy()
            np.save(path, val)
            del val

    def _weight_loader(self, param_name):
        safe_name = param_name.replace("/", "_").replace(":", "_")
        path = os.path.join(self.temp_dir, safe_name + ".npy")
        if os.path.exists(path):
            return np.load(path, mmap_mode='r')
        return None

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
                if isinstance(optimizer, str):
                    shard_opt = keras.optimizers.get(optimizer)
                else:
                    shard_opt = optimizer.from_config(optimizer.get_config())
                
                with keras.device(self.devices[i]):
                    shard.compile(optimizer=shard_opt, **kwargs)
        else:
            super().compile(optimizer=optimizer, **kwargs)