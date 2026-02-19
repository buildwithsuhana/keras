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
    print("Creating OPT backbone under distribution scope...")

# Import KerasHub to get OPT
import keras_hub

with distribution.scope():
    backbone = keras_hub.models.OPTBackbone.from_preset("opt_125m_en")
    # Wrap in a functional model to use standard fit
    model = keras.Model(backbone.inputs, backbone(backbone.inputs))

# Verify sharding
if rank == 0:
    print("\nVerifying weight sharding (first few weights):")
    for variable in model.weights[:5]:
        val = variable.value
        from torch.distributed.tensor import DTensor
        is_dtensor = isinstance(val, DTensor)
        sharding = val.placements if is_dtensor else "None"
        print(f"Variable {variable.path}: is_dtensor={is_dtensor}, shape={val.shape}, sharding={sharding}")

# fit test
if rank == 0:
    print("\nRunning fit test...")

# Create dummy data
batch_size = 2 * world_size
seq_len = 32
# OPT backbone expects "token_ids" and "padding_mask"
x = {
    "token_ids": np.random.randint(0, 50272, (batch_size, seq_len)).astype("int32"),
    "padding_mask": np.ones((batch_size, seq_len)).astype("int32"),
}
y = np.random.randn(batch_size, seq_len, 768).astype("float32")

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss="mse",
)

# Run fit
model.fit(x, y, epochs=1, batch_size=batch_size)

if rank == 0:
    print("\nFit test completed successfully!")
