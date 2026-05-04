import os
os.environ["KERAS_BACKEND"] = "jax"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["KERAS_BACKEND_DEVICE"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

import sys
import numpy as np
import json
import keras
import keras_hub

if __name__ == "__main__":
    # Disable Dropout for consistency
    keras.config.disable_interactive_logging()
    keras.utils.set_random_seed(42)

    world_size = 2
    devices = keras.distribution.list_devices("cpu")
    if len(devices) < world_size:
        devices = ["cpu:0"] * world_size
    else:
        devices = devices[:world_size]

    mesh = keras.distribution.DeviceMesh(
        shape=(world_size,),
        axis_names=("model",),
        devices=devices,
    )

    layout_map = keras.distribution.LayoutMap(mesh)
    layout_map["embeddings/token_embedding/.*"] = keras.distribution.TensorLayout((None, None), mesh)
    layout_map["embeddings/position_embedding/.*"] = keras.distribution.TensorLayout((None, None), mesh)
    layout_map[".*attention.*query.*kernel"] = keras.distribution.TensorLayout(("model", None), mesh)
    layout_map[".*attention.*key.*kernel"] = keras.distribution.TensorLayout(("model", None), mesh)
    layout_map[".*attention.*value.*kernel"] = keras.distribution.TensorLayout(("model", None), mesh)

    distribution = keras.distribution.ModelParallel(
        layout_map=layout_map,
        batch_dim_name="model",
        auto_shard_dataset=False,
    )

    with distribution.scope():
        model = keras_hub.models.OPTBackbone.from_preset("opt_125m_en")
        model.load_weights("initial_weights.weights.h5")

        data = np.load("test_data.npz")
        x_tokens = data["x_tokens"][:2]
        x_mask = data["x_mask"][:2]
        
        x = {"token_ids": x_tokens, "padding_mask": x_mask}

        output = model(x, training=False)
        output_np = keras.ops.convert_to_numpy(output)
        
        print(f"OUTPUT_SAMPLE: {output_np[0].flatten()[:5].tolist()}")
