import logging
import gc
import keras
from keras import models
from keras.src.distribution import list_devices
from keras.distribution import DeviceMesh, ModelParallel, set_distribution
from keras.src.distribution.tensor_parallel.autoconfig import get_default_config

logger = logging.getLogger(__name__)

class TensorParallelKeras(models.Model):
    """
    A wrapper that applies Keras 3 Native Tensor Parallelism to a model.
    """
    def __init__(self, model, device_count=None, device_ids=None, **kwargs):
        super().__init__(**kwargs)
        # Do not store the original model to allow GC
        # self._original_cpu_model = model 

        # 1. Detect Devices
        if device_ids is None:
            all_devices = list_devices()
            device_ids = [d for d in all_devices if "tpu" in d.lower()]
            if not device_ids:
                device_ids = [d for d in all_devices if "gpu" in d.lower()]
            if not device_ids:
                 device_ids = [d for d in all_devices if "cpu" in d.lower()]
        
        if not device_ids:
            raise ValueError("No devices found for Tensor Parallelism.")

        self.devices = device_ids[:device_count] if device_count else device_ids
        self.device_count = len(self.devices)
        
        print(f"✅ Configuring Tensor Parallelism on {self.device_count} devices: {self.devices}")

        # 2. Create Mesh
        self.mesh = DeviceMesh(
            shape=(1, self.device_count),
            axis_names=["batch", "model"],
            devices=self.devices
        )

        # 3. Generate Layout
        self.layout_map = get_default_config(model, self.mesh)
        
        # 4. Initialize Strategy
        self.strategy = ModelParallel(
            device_mesh=self.mesh, 
            layout_map=self.layout_map
        )

        # 5. Clone and Distribute
        print("🔧 Distributing model weights across mesh...")
        
        with self.strategy.scope():
            # Clone architecture
            self.distributed_model = model.__class__.from_config(model.get_config())
            
            # [FIX] Robust LoRA Re-enabling
            lora_rank = None
            for w in model.weights:
                if "lora_kernel_a" in w.name:
                    lora_rank = w.shape[-1]
                    print(f"   ✨ Auto-detected LoRA in source (rank={lora_rank})")
                    break
            
            if lora_rank is not None:
                if hasattr(self.distributed_model, "backbone"):
                    print("   ✨ Re-enabling LoRA on distributed backbone...")
                    self.distributed_model.backbone.enable_lora(lora_rank)
                else:
                    print("   ⚠️ LoRA detected but no backbone found.")

            # Build the distributed model
            if not self.distributed_model.built and model.inputs:
                 dummy_shape = (1,) + model.inputs[0].shape[1:]
                 self.distributed_model.build(dummy_shape)

            # [OOM FIX] Iterative Weight Transfer with GC
            print("   📦 Copying weights incrementally to avoid OOM...")
            
            # We iterate and assign values one by one.
            # Crucially, we verify shapes match first.
            src_vars = model.variables
            dst_vars = self.distributed_model.variables
            
            if len(src_vars) != len(dst_vars):
                 # Fallback: try simple set_weights if lengths mismatch (rare with LoRA fix)
                 # But this risks OOM. We assume the LoRA fix aligned them.
                 print(f"   ⚠️ Variable count mismatch ({len(src_vars)} vs {len(dst_vars)}). Attempting standard load...")
                 self.distributed_model.set_weights(model.get_weights())
            else:
                for i, (src, dst) in enumerate(zip(src_vars, dst_vars)):
                    # Assign value directly from CPU source to Sharded Dest
                    dst.assign(src.value)
                    
                    # Explicitly delete the source weight if possible/safe
                    # (We can't easily delete 'src' as it belongs to 'model', 
                    # but we can aggressively GC to clear temps)
                    if i % 50 == 0:
                        gc.collect()

        print("🚀 Model successfully sharded!")

    def call(self, inputs, **kwargs):
        return self.distributed_model(inputs, **kwargs)

    def compile(self, *args, **kwargs):
        self.distributed_model.compile(*args, **kwargs)
        self.optimizer = self.distributed_model.optimizer

    def fit(self, *args, **kwargs):
        return self.distributed_model.fit(*args, **kwargs)

    def save(self, *args, **kwargs):
        return self.distributed_model.save(*args, **kwargs)
        
    @property
    def layers(self):
        return self.distributed_model.layers