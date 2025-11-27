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
        super().__init__(**kwargs)
        
        # 1. Detect Devices
        if device_ids is None:
            all_devices = list_devices()
            device_ids = [d for d in all_devices if "tpu" in d.lower()]
            if not device_ids:
                device_ids = [d for d in all_devices if "gpu" in d.lower()]
            if not device_ids:
                 device_ids = [d for d in all_devices if "cpu" in d.lower()]
        
        self.devices = device_ids[:device_count] if device_count else device_ids
        self.device_count = len(self.devices)
        self.rank = rank
        
        if self.devices and self.rank < len(self.devices):
            self.current_device = self.devices[self.rank]
        else:
            self.current_device = "cpu"

        print(f"✅ [TP Init] Rank {self.rank}/{self.device_count} -> Device: {self.current_device}")

        # 2. Create Logical Mesh
        self.mesh = DeviceMesh(
            shape=(1, self.device_count),
            axis_names=["batch", "model"],
            devices=self.devices
        )

        # 3. Generate Layout Rules
        self.layout_map = get_default_config(model, self.mesh)
        
        # 4. Initialize Manual Strategy
        self.strategy = ParameterShardingStrategy(
            device_count=self.device_count, 
            rank=self.rank
        )

        # 5. Shard the model
        self.distributed_model, _ = self.strategy.shard_model_parameters(
            model, 
            self.layout_map, 
            device_id=self.current_device
        )

        print("🚀 [TP Init] Model successfully manually sharded!")

    def call(self, inputs, **kwargs):
        return self.distributed_model(inputs, **kwargs)

    def compile(self, *args, **kwargs):
        self.distributed_model.compile(*args, **kwargs)
        if hasattr(self.distributed_model, "optimizer"):
            self.optimizer = self.distributed_model.optimizer

    def fit(self, *args, **kwargs):
        return self.distributed_model.fit(*args, **kwargs)

    @property
    def layers(self):
        return self.distributed_model.layers