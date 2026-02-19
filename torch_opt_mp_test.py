import os

os.environ["KERAS_BACKEND"] = "torch"

import torch
import torch.distributed as dist
import numpy as np
import keras
from keras.src.distribution import DeviceMesh, LayoutMap, ModelParallel

# Initialize distribution
if "RANK" not in os.environ:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"

keras.distribution.initialize()

world_size = int(os.environ.get("WORLD_SIZE", "1"))
rank = int(os.environ.get("RANK", "0"))

# Create a 2D mesh for testing
mesh = DeviceMesh(shape=(1, world_size), axis_names=("batch", "model"))

layout_map = LayoutMap(mesh)
layout_map["token_embedding/embeddings"] = (None, "model")
layout_map["decoder_block_.*_attention/query/kernel"] = (None, "model")
layout_map["decoder_block_.*_attention/key/kernel"] = (None, "model")
layout_map["decoder_block_.*_attention/value/kernel"] = (None, "model")
layout_map["decoder_block_.*_attention/output_dense/kernel"] = ("model", None)
layout_map["decoder_block_.*_ffn_layers_0/kernel"] = (None, "model")
layout_map["decoder_block_.*_ffn_layers_1/kernel"] = ("model", None)

distribution = ModelParallel(
    layout_map=layout_map, batch_dim_name="batch"
)
keras.distribution.set_distribution(distribution)

if rank == 0:
    print(f"World size: {world_size}")
    print("Creating OPT model under distribution scope...")

# Import KerasHub to get OPT
import keras_hub

with distribution.scope():
    model = keras_hub.models.OPTCausalLM.from_preset("opt_125m_en")

# Verify sharding
if rank == 0:
    print("\nVerifying weight sharding (first few weights):")
    for variable in model.weights[:5]:
        val = variable.value
        from torch.distributed.tensor import DTensor
        is_dtensor = isinstance(val, DTensor)
        sharding = val.placements if is_dtensor else "None"
        print(f"Variable {variable.path}: is_dtensor={is_dtensor}, shape={val.shape}, sharding={sharding}")

# Generate test
if rank == 0:
    print("\nRunning generation test...")

prompt = "Keras is"
output = model.generate(prompt, max_length=20)

if rank == 0:
    print(f"\nGenerated output: {output}")
