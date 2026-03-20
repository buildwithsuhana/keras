import os

# Set backend to torch before anything else
os.environ["KERAS_BACKEND"] = "torch"

import torch
import numpy as np
import keras
import keras_hub
from keras.src.distribution.distribution_lib import (
    DeviceMesh,
    DataParallel,
)
from keras.src.backend.torch import distribution_lib as torch_distribution_lib
from keras.src.backend.torch.core import convert_to_tensor

def test_data_parallel():
    # 1. Initialize PyTorch Distributed Environment
    keras.distribution.initialize()
    
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    
    if rank == 0:
        print(f"Starting DataParallel test with {world_size} processes using OPT 125M...")

    # 2. Define the Distribution Strategy
    device_type = "gpu"
    devices = keras.distribution.list_devices(device_type)
    
    # For DataParallel, we just need the devices
    distribution = DataParallel(devices=devices)

    # 3. Build Model within the Distribution Scope
    with distribution.scope():
        # Load the OPT model
        model = keras_hub.models.OPTBackbone.from_preset("opt_125m_en")
        
        # Simple loss and optimizer for testing
        model.compile(
            optimizer="adam",
            loss="mse",
        )

        if rank == 0:
            print("Model built and compiled.")

        # 4. Prepare Input Data
        # We shard the data manually for this test.
        global_batch_size = 8
        per_process_batch_size = global_batch_size // world_size
        seq_len = 32
        
        # Ensure each rank gets different data if we want to be realistic, 
        # but for simple test, same data is fine as long as we shard.
        x_local = {
            "token_ids": np.random.randint(0, 50272, size=(per_process_batch_size, seq_len)).astype("int32"),
            "padding_mask": np.ones((per_process_batch_size, seq_len), dtype="int32"),
        }
        y_local = np.random.randn(per_process_batch_size, seq_len, 768).astype("float32")
        
        # 5. Distribute data
        def _to_dtensor(v):
            if isinstance(v, dict):
                return {k: _to_dtensor(val) for k, val in v.items()}
            if isinstance(v, (np.ndarray, torch.Tensor)):
                t = convert_to_tensor(v)
                return torch_distribution_lib.distribute_data_input(
                    t, distribution.get_data_layout(t.shape), distribution.batch_dim_name
                )
            return v

        x_dtensor = _to_dtensor(x_local)
        y_dtensor = _to_dtensor(y_local)

        if rank == 0:
            print("Data prepared.")

        # 6. Run Training
        # Use a generator to provide data for fit()
        def _gen():
            while True:
                yield x_dtensor, y_dtensor

        if rank == 0:
            print("Starting model.fit()...")

        history = model.fit(
            _gen(),
            steps_per_epoch=2,
            epochs=3,
            verbose=1,
        )
        
        loss = history.history["loss"][-1]
        if rank == 0:
            print(f"Training completed. Final loss: {loss}")

        # 7. Verification
        # Check if weights are regular torch.nn.Parameter (NOT DTensor)
        for weight in model.weights:
            if isinstance(weight.value, torch_distribution_lib.DTensor):
                raise ValueError(f"Weight {weight.name} should NOT be a DTensor in DataParallel")
            if not isinstance(weight.value, torch.nn.Parameter):
                raise ValueError(f"Weight {weight.name} is not a torch.nn.Parameter")

        # Verify DDP model was created
        if model._ddp_model is None:
            raise ValueError("DDP model was not initialized")

        if rank == 0:
            print("Verification successful: Weights are regular Parameters and DDP is initialized.")

if __name__ == "__main__":
    test_data_parallel()
