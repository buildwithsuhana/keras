import logging
import gc
import keras
from keras.src.distribution.tensor_parallel.autoconfig import get_default_config
from keras.src.distribution.tensor_parallel.parameter_sharding import make_parameter_sharded_model
from keras.src.distribution.tensor_parallel.coordinated_optimizer import TensorParallelOptimizer
from keras.src.distribution import list_devices
from keras.src.models import Model

logger = logging.getLogger(__file__)

class TensorParallelKeras(Model):
    def __init__(
        self,
        model, # The Master Model (Should be on CPU)
        device_count=None,
        device_ids=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Device Setup
        if device_count is None or device_ids is None:
            # Fallback auto-detection
            all_devices = list_devices()
            device_count = len(all_devices)
            device_ids = [str(d) for d in all_devices]

        self.device_count = device_count
        self.devices = device_ids
        
        # Normalize device IDs
        self.devices = [d if "gpu" in d or "cpu" in d or "tpu" in d else f"gpu:{d}" for d in self.devices]

        # Get Sharding Config
        self.tensor_parallel_config = get_default_config(model, self.devices)
        self.model_shards = []
        
        print(f"🚀 Initializing Tensor Parallelism on {self.devices}")

        # --- Lazy Sharding Loop ---
        for rank, device_id in enumerate(self.devices):
            print(f"[{device_id}] ⏳ creating shard {rank+1}/{self.device_count}...")
            
            # 1. Create Skeleton on Target Device
            # We clone the structure. Weights are initialized (randomly) on the GPU here.
            # Since we slice immediately after, this temporary random memory is overwritten.
            with keras.device(device_id):
                shard = keras.models.clone_model(model)
                # Propagate mixed precision policy
                if hasattr(model, 'dtype_policy'):
                    shard.dtype_policy = model.dtype_policy
            
            # Build if needed to create variables
            if not shard.built and model.inputs:
                try:
                    shard.build([x.shape for x in model.inputs])
                except Exception:
                    pass

            # 2. Slice & Copy (CPU -> GPU)
            # This function reads from 'model' (CPU), slices it, and assigns to 'shard' (GPU)
            shard, _ = make_parameter_sharded_model(
                shard_model=shard,
                source_model=model,
                config=self.tensor_parallel_config,
                rank=rank,
                device_count=self.device_count,
                device_id=device_id,
            )

            self.model_shards.append(shard)
            
            # 3. Clean up CPU Intermediates
            gc.collect()
            print(f"[{device_id}] ✅ Shard created.")

        # --- Cleanup Master ---
        print("🗑️  Freeing Master Model from CPU...")
        self._original_model = None
        # We cannot 'del' the argument passed in, but we drop our reference.
        # The caller should also delete their reference if possible.
        gc.collect()

        self.built = True
        self.distributed = True
        self.assembled_model = self.build_assembled_model()

    def build_assembled_model(self):
        """Virtual graph combining shards."""
        # Use first shard for input specs
        ref = self.model_shards[0]
        inputs = {
            i.name.split(':')[0]: keras.Input(shape=i.shape[1:], dtype=i.dtype, name=i.name.split(':')[0])
            for i in ref.inputs
        }
        
        shard_outputs = []
        for shard in self.model_shards:
            # Map global inputs to shard inputs
            shard_in = {
                k: inputs[k.split(':')[0]] for k in [x.name for x in shard.inputs] 
                if k.split(':')[0] in inputs
            }
            if not shard_in: shard_in = inputs # fallback
            shard_outputs.append(shard(shard_in))
        
        # Combine outputs (Sum logits)
        if len(shard_outputs) > 1:
            out = keras.layers.Add()(shard_outputs)
        else:
            out = shard_outputs[0]
            
        return keras.Model(inputs=inputs, outputs=out)

    def call(self, inputs, training=None, **kwargs):
        return self.assembled_model(inputs, training=training, **kwargs)

    def compile(self, optimizer=None, **kwargs):
        if len(self.model_shards) > 1 and optimizer:
            # Wrap optimizer
            opt = TensorParallelOptimizer(optimizer, self.device_count)
            opt._shard_models = self.model_shards
            
            # Build Var Map
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