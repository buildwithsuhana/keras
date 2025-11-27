import logging
import keras
from keras import models
from keras.src.distribution import list_devices
from keras.distribution import DeviceMesh, ModelParallel
from keras.src.distribution.tensor_parallel.autoconfig import get_default_config

logger = logging.getLogger(__name__)

class TensorParallelKeras(models.Model):
    """
    A wrapper that applies Keras 3 Native Tensor Parallelism to a model.
    """
    def __init__(self, model, device_count=None, device_ids=None, **kwargs):
        super().__init__(**kwargs)
        self._original_cpu_model = model

        # 1. Detect Devices
        if device_ids is None:
            all_devices = list_devices()
            # Prefer TPUs, then GPUs
            device_ids = [d for d in all_devices if "tpu" in d.lower()]
            if not device_ids:
                device_ids = [d for d in all_devices if "gpu" in d.lower()]
            # Fallback to CPU for testing
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

        # 3. Generate Layout (Sharding Config)
        self.layout_map = get_default_config(model, self.mesh)
        
        # 4. Initialize Distribution Strategy
        self.strategy = ModelParallel(
            device_mesh=self.mesh, 
            layout_map=self.layout_map
        )

        # 5. Clone and Distribute
        print("🔧 Distributing model weights across mesh...")
        
        # Use .scope() to create the distributed model on the devices
        with self.strategy.scope():
            # Clone architecture (Note: this creates a clean model without LoRA)
            self.distributed_model = model.__class__.from_config(model.get_config())
            
            # [FIX] Robustly detect and re-enable LoRA
            # We inspect the source weights. If we see 'lora_kernel_a', we know LoRA is on.
            # We infer the rank from the shape of the 'a' kernel (typically [dim, rank]).
            lora_rank = None
            for w in model.weights:
                if "lora_kernel_a" in w.name:
                    # KerasHub LoRA 'A' kernel shape is (input_dim, rank)
                    lora_rank = w.shape[-1]
                    print(f"   ✨ Auto-detected LoRA in source model (rank={lora_rank})")
                    break
            
            if lora_rank is not None:
                # Re-apply LoRA to the distributed model so shapes match
                if hasattr(self.distributed_model, "backbone"):
                    print("   ✨ Re-enabling LoRA on distributed backbone...")
                    self.distributed_model.backbone.enable_lora(lora_rank)
                else:
                    print("   ⚠️ LoRA detected but could not find 'backbone' to enable it.")

            # Build if needed
            if not self.distributed_model.built and model.inputs:
                 dummy_shape = (1,) + model.inputs[0].shape[1:]
                 self.distributed_model.build(dummy_shape)

            # Copy weights (Now the structures should match!)
            self.distributed_model.set_weights(model.get_weights())

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