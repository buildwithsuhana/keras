import os
import sys

# Set environment variables BEFORE importing keras
os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
os.environ["JAX_PLATFORMS"] = "cpu"

import json
import keras
from shared_utils import get_layout_map, get_data, train_model

def run_jax(world_size):
    keras.utils.set_random_seed(42)
    # Ensure we use CPU devices
    devices = keras.distribution.list_devices("cpu")[:world_size]
    mesh = keras.distribution.DeviceMesh((world_size,), ["model"], devices)
    dist = keras.distribution.ModelParallel(layout_map=get_layout_map(mesh), batch_dim_name="model")
    
    x, y = get_data()
    loss = train_model(dist, x, y)
    with open("results_jax.json", "w") as f:
        json.dump({"final_loss": loss}, f)

if __name__ == "__main__":
    run_jax(world_size=2)
