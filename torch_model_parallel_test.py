import os

os.environ["KERAS_BACKEND"] = "torch"

import torch
import torch.distributed as dist
import numpy as np
import keras
keras.config.disable_traceback_filtering()
from keras.distribution import DeviceMesh, LayoutMap, ModelParallel

# Initialize distribution
if "RANK" not in os.environ:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["GLOO_SOCKET_IFNAME"] = "lo0"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"

keras.distribution.initialize()

world_size = int(os.environ.get("WORLD_SIZE", "1"))
rank = int(os.environ.get("RANK", "0"))

# Create a 2D mesh for testing
mesh = DeviceMesh(shape=(1, world_size), axis_names=("batch", "model"))

layout_map = LayoutMap(mesh)
layout_map["dense_1/kernel"] = (None, "model")
layout_map["dense_2/kernel"] = ("model", None)

distribution = ModelParallel(
    layout_map=layout_map, batch_dim_name="batch"
)
keras.distribution.set_distribution(distribution)

if rank == 0:
    print(f"World size: {world_size}")
    print("Creating model under distribution scope...")

with distribution.scope():
    model = keras.Sequential([
        keras.layers.Dense(64, activation="relu", name="dense_1", input_shape=(32,)),
        keras.layers.Dense(32, name="dense_2")
    ])

# Verify sharding
if rank == 0:
    print("\nVerifying weight sharding:")
    for variable in model.weights:
        val = variable.value
        from torch.distributed.tensor import DTensor
        is_dtensor = isinstance(val, DTensor)
        sharding = val.placements if is_dtensor else "None"
        print(f"Variable {variable.path}: is_dtensor={is_dtensor}, shape={val.shape}, sharding={sharding}")

# fit test
if rank == 0:
    print("\nRunning fit test...")

# Create dummy data
batch_size = 4
x = np.random.randn(batch_size, 32).astype("float32")
y = np.random.randn(batch_size, 32).astype("float32")

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="mse",
)

# Run fit
model.fit(x, y, epochs=2, batch_size=batch_size)

if rank == 0:
    print("\nFit test completed successfully!")
