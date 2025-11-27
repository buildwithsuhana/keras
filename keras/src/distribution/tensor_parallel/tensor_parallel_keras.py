import logging
import gc
import keras
from keras import models
from keras.src.distribution import list_devices
from keras.distribution import DeviceMesh

# Import the custom Manual Strategy & Config
from keras.src.distribution.tensor_parallel.parameter_sharding import ParameterShardingStrategy
from keras.src.distribution.tensor_parallel.autoconfig import get_default_config

logger = logging.getLogger(__name__)

class TensorParallelKeras(models.Model):
    def __init__(self, model, device_count=None, device_ids=None, rank=0, **kwargs):
        """
        Args:
            model: The source Keras model to be sharded.
            device_count: Total number of devices involved in TP.
            device_ids: List of device strings (optional auto-detection).
            rank: The global rank of the current process/device (0 to device_count-1).
        """
        super().__init__(**kwargs)
        
        # 1. Detect Devices
        if device_ids is None:
            all_devices = list_devices()
            # Prioritize TPU, then GPU
            device_ids = [d for d in all_devices if "tpu" in d.lower()]
            if not device_ids:
                device_ids = [d for d in all_devices if "gpu" in d.lower()]
            if not device_ids:
                 device_ids = [d for d in all_devices if "cpu" in d.lower()]
        
        self.devices = device_ids[:device_count] if device_count else device_ids
        self.device_count = len(self.devices)
        self.rank = rank
        
        # Pick the specific device for this rank (e.g. "gpu:0" or "tpu:0")
        if self.devices and self.rank < len(self.devices):
            self.current_device = self.devices[self.rank]
        else:
            self.current_device = "cpu"

        print(f"✅ TP Setup: Rank {self.rank}/{self.device_count} targeting device: {self.current_device}")

        # 2. Create Logical Mesh
        # Used by autoconfig to generate rules (even if we execute manually)
        self.mesh = DeviceMesh(
            shape=(1, self.device_count),
            axis_names=["batch", "model"],
            devices=self.devices
        )

        # 3. Generate Layout Rules
        # Uses the robust graph traversal from autoconfig.py
        self.layout_map = get_default_config(model, self.mesh)
        
        # 4. Initialize Manual Strategy
        # We use the custom ParameterShardingStrategy for OOM-safe loading
        self.strategy = ParameterShardingStrategy(
            device_count=self.device_count, 
            rank=self.rank
        )

        # 5. Shard the model
        # This will:
        #   a. Convert weights to NumPy (CPU)
        #   b. Slice them for this rank
        #   c. Move shards to GPU/TPU immediately
        #   d. Delete CPU copies to prevent RAM OOM
        self.distributed_model, _ = self.strategy.shard_model_parameters(
            model, 
            self.layout_map, 
            device_id=self.current_device
        )

        print("🚀 Model successfully manually sharded!")

    def call(self, inputs, **kwargs):
        # Delegates forward pass to the wrapped sharded model
        return self.distributed_model(inputs, **kwargs)

    def compile(self, *args, **kwargs):
        self.distributed_model.compile(*args, **kwargs)
        # Expose optimizer for external access if needed
        if hasattr(self.distributed_model, "optimizer"):
            self.optimizer = self.distributed_model.optimizer

    def fit(self, *args, **kwargs):
        return self.distributed_model.fit(*args, **kwargs)

    @property
    def layers(self):
        return self.distributed_model.layers