import os

# Set backend to torch before anything else
os.environ["KERAS_BACKEND"] = "torch"

import torch
import numpy as np
import keras
import keras_hub
from keras.src.distribution.distribution_lib import (
    DeviceMesh,
    LayoutMap,
    ModelParallel,
)

def test_model_parallel():
    # 1. Initialize PyTorch Distributed Environment
    keras.distribution.initialize()
    
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    
    if rank == 0:
        print(f"Starting ModelParallel test with {world_size} processes using OPT 125M...")

    # 2. Define the Distribution Strategy
    # We use 'cpu' for testing to avoid needing multiple GPUs
    device_type = "cpu"
    devices = keras.distribution.list_devices(device_type)
    
    # Create a 1D mesh for model parallelism
    mesh = DeviceMesh(
        shape=(world_size,), 
        axis_names=("model",), 
        devices=devices
    )
    
    # Simple LayoutMap: we can leave it empty to test default replication
    # or add specific rules. Here we test the infrastructure.
    layout_map = LayoutMap(mesh)
    distribution = ModelParallel(layout_map=layout_map, batch_dim_name="model")

    # 3. Build Model within the Distribution Scope
    with distribution.scope():
        # Load a small OPT model
        model = keras_hub.models.OPTBackbone.from_preset("opt_125m_en")
        
        model.compile(
            optimizer="adam",
            loss="mse",
        )

        if rank == 0:
            print("Model built and compiled.")

        # 4. Prepare Regular NumPy Data
        # The trainer will automatically distribute this data as DTensors
        batch_size = 4 * world_size
        seq_len = 32
        x = {
            "token_ids": np.random.randint(0, 50272, size=(batch_size, seq_len)).astype("int32"),
            "padding_mask": np.ones((batch_size, seq_len), dtype="int32"),
        }
        y = np.random.randn(batch_size, seq_len, 768).astype("float32")

        if rank == 0:
            print("Data prepared. Starting model.fit()...")

        # 5. Run Training
        # We can now pass raw arrays directly! 
        # Keras will shard them (DataParallel-style for inputs) and 
        # wrap them in DTensors for the ModelParallel model.
        history = model.fit(
            x, y,
            epochs=2,
            batch_size=batch_size,
            verbose=1 if rank == 0 else 0,
        )

        loss = history.history["loss"][-1]
        if rank == 0:
            print(f"SUCCESS: Training completed. Final loss: {loss}")

if __name__ == "__main__":
    try:
        test_model_parallel()
    except Exception as e:
        print(f"Test failed on rank {os.environ.get('RANK', 'unknown')}: {e}")
        import traceback
        traceback.print_exc()
        # Ensure we exit with error code so CI/torchrun detects failure
        os._exit(1)
