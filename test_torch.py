import os
os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["KERAS_BACKEND_DEVICE"] = "cpu"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import sys
import numpy as np
import json
import keras
import keras_hub
import torch
import torch.multiprocessing as mp

# Disable MPS globally
if hasattr(torch.backends, "mps"):
    torch.backends.mps.is_available = lambda: False
if hasattr(torch, "set_default_device"):
    torch.set_default_device("cpu")

def _worker(rank, world_size, return_dict):
    import torch
    if hasattr(torch.backends, "mps"):
        torch.backends.mps.is_available = lambda: False
    if hasattr(torch, "set_default_device"):
        torch.set_default_device("cpu")
        
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29570"
    
    # Disable Dropout for consistency
    keras.config.disable_interactive_logging()
    keras.utils.set_random_seed(42)

    keras.distribution.initialize()

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
        
        samples_per_rank = len(x_tokens) // world_size
        start = rank * samples_per_rank
        end = (rank + 1) * samples_per_rank
        x = {"token_ids": x_tokens[start:end], "padding_mask": x_mask[start:end]}

        output = model(x, training=False)
        output_np = keras.ops.convert_to_numpy(output)
        
        if rank == 0:
            return_dict["output"] = output_np.flatten()[:5].tolist()

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    manager = mp.Manager()
    return_dict = manager.dict()
    mp.spawn(_worker, args=(2, return_dict), nprocs=2, join=True)
    if "output" in return_dict:
        print(f"OUTPUT_SAMPLE: {return_dict['output']}")
    else:
        print("Worker failed to produce output")
