import logging
import gc
import keras
from keras import models
from keras.src.distribution import list_devices
from keras.distribution import DeviceMesh

# Import the custom Manual Strategy
from keras.src.distribution.tensor_parallel.parameter_sharding import ParameterShardingStrategy
from keras.src.distribution.tensor_parallel.autoconfig import get_default_config

logger = logging.getLogger(__name__)

class TensorParallelKeras(models.Model):
    def __init__(self, model, device_count=None, device_ids=None, rank=0, **kwargs):
        """
        Args:
            model: The source model.
            device_count: Total devices.
            rank: The rank of the current device (default 0 for single-device simulation).
        """
        super().__init__(**kwargs)
        
        # 1. Detect Devices
        if device_ids is None:
            all_devices = list_devices()
            device_ids = [d for d in all_devices if "tpu" in d.lower()]
            if not device_ids:
                device_ids = [d for d in all_devices if "gpu" in d.lower()]
        
        self.devices = device_ids[:device_count] if device_count else device_ids
        self.device_count = len(self.devices)
        self.rank = rank
        
        # Pick the specific device for this rank
        self.current_device = self.devices[self.rank] if self.devices else "cpu"

        print(f"✅ TP Setup: Rank {self.rank}/{self.device_count} on {self.current_device}")

        # 2. Create Mesh (Logical)
        self.mesh = DeviceMesh(
            shape=(1, self.device_count),
            axis_names=["batch", "model"],
            devices=self.devices
        )

        # 3. Generate Layout (Using fixed LayoutMap class)
        self.layout_map = get_default_config(model, self.mesh)
        
        # 4. Initialize Manual Strategy
        # We DO NOT use ModelParallel class. We use ParameterShardingStrategy.
        self.strategy = ParameterShardingStrategy(
            device_count=self.device_count, 
            rank=self.rank
        )

        # 5. Shard the model
        # This will internally convert weights to numpy, slice them, 
        # and create a new model with only the shards for THIS rank.
        self.distributed_model, _ = self.strategy.shard_model_parameters(
            model, 
            self.layout_map, 
            device_id=self.current_device
        )

        print("🚀 Model successfully manually sharded!")

    def call(self, inputs, **kwargs):
        return self.distributed_model(inputs, **kwargs)

    def compile(self, *args, **kwargs):
        self.distributed_model.compile(*args, **kwargs)

    def fit(self, *args, **kwargs):
        return self.distributed_model.fit(*args, **kwargs)