import os

os.environ["KERAS_BACKEND"] = "torch"

import torch
import torch.distributed as dist
import numpy as np
import keras
import keras_hub
from keras.distribution import DeviceMesh, LayoutMap, ModelParallel

# Initialize distribution for single rank testing
if "RANK" not in os.environ:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"

keras.distribution.initialize()

# Create a 1D mesh for testing (since I only have 1 rank easily available)
mesh = DeviceMesh(shape=(1,), axis_names=("model",), devices=["cpu:0"])

layout_map = LayoutMap(mesh)
# Shard ONLY the feedforward weights which are simpler
layout_map["feedforward_intermediate_dense/kernel"] = (None, "model")
layout_map["feedforward_output_dense/kernel"] = ("model", None)

distribution = ModelParallel(layout_map=layout_map, batch_dim_name=None)
keras.distribution.set_distribution(distribution)

print("Loading model under distribution scope...")
model = keras_hub.models.OPTCausalLM.from_preset("opt_125m_en")

# Verify sharding
print("\nVerifying weight sharding:")
sharded_count = 0
for variable in model.weights:
    layout = distribution.get_variable_layout(variable)
    if layout is not None and any(axis is not None for axis in layout.axes):
        sharded_count += 1
        val = variable.value
        from torch.distributed.tensor import DTensor
        is_dtensor = isinstance(val, DTensor)
        print(f"Variable {variable.path}: sharded={True}, is_dtensor={is_dtensor}")

print(f"\nTotal sharded variables: {sharded_count}")

# Simple forward pass
print("\nRunning generation test...")
# Use a very short input to speed up
inputs = ["The quick"]
outputs = model.generate(inputs, max_length=5)
for i, out in enumerate(outputs):
    print(f"Generated {i}: {out}")

print("\nTest completed successfully!")
