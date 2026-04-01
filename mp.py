import os
import random
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.distributed as dist

# Set backend to torch
os.environ["KERAS_BACKEND"] = "torch"
# Force CPU for local testing if no GPU is available
if not torch.cuda.is_available():
    os.environ["KERAS_TORCH_DEVICE"] = "cpu"

import keras
import keras_hub
from keras.src.distribution import DeviceMesh, LayoutMap, ModelParallel

def worker(rank, world_size, port):
    # 1. Setup Environment
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)

    from keras.src.backend.torch import distribution_lib as torch_dist_lib
    
    # 2. Initialize Distributed Backend
    torch_dist_lib.initialize()
    print(f"Process {rank} initialized.")

    # 3. Define Device Mesh (1 Batch Dim, 2 Model Shards)
    mesh = DeviceMesh(
        shape=(1, world_size), 
        axis_names=("batch", "model"), 
        devices=[f"cpu:{i}" for i in range(world_size)] if not torch.cuda.is_available() else None
    )

    # 4. Define Layout Map for OPT
    # Shard the vocabulary (embedding) and the feed-forward layers
    layout_map = LayoutMap(mesh)
    # Note: Using regex to match Keras Hub parameter names
    layout_map[".*token_embedding/embeddings"] = (None, "model")
    layout_map[".*query/kernel"] = (None, "model")
    layout_map[".*key/kernel"] = (None, "model")
    layout_map[".*value/kernel"] = (None, "model")
    layout_map[".*attention_output/kernel"] = ("model", None)
    layout_map[".*intermediate_output/kernel"] = (None, "model")
    layout_map[".*layer_output/kernel"] = ("model", None)

    # 5. Create ModelParallel Distribution
    distribution = ModelParallel(
        layout_map=layout_map, 
        batch_dim_name="batch"
    )

    # 6. Load Model within Distribution Scope
    print(f"Process {rank}: Loading OPT-125M...")
    with distribution.scope():
        # Loading the backbone
        model = keras_hub.models.OPTBackbone.from_preset("opt_125m")
    
    # 7. Verify Sharding
    from torch.distributed.tensor import DTensor
    
    # Check the first transformer layer's query kernel
    # OPTBackbone -> transformer_layers (list) -> _self_attention_layer -> _query_dense -> kernel
    layer = model.transformer_layers[0]
    q_kernel = layer._self_attention_layer._query_dense.kernel
    print(f"Process {rank}: Query kernel shape: {q_kernel.shape}")
    
    if isinstance(q_kernel.value.data, DTensor):
        print(f"Process {rank}: SUCCESS - Query kernel is sharded as DTensor.")
        placements = q_kernel.value.data.placements
        print(f"Process {rank}: Placements: {placements}")
    else:
        print(f"Process {rank}: FAILURE - Query kernel is a regular tensor.")

    # 8. Run Forward Pass
    print(f"Process {rank}: Running forward pass...")
    # Generate dummy input (batch_size=1, seq_len=8)
    # Use torch.long for token IDs
    input_ids = torch.randint(0, 1000, (1, 8)).to(torch.long)
    padding_mask = torch.ones((1, 8)).to(torch.long)
    
    input_data = {
        "token_ids": input_ids,
        "padding_mask": padding_mask,
    }
    
    # Distribute input data (batch parallel)
    input_data = {
        k: torch_dist_lib.distribute_data_input(
            v, distribution.get_data_layout(v.shape), distribution.batch_dim_name
        ) for k, v in input_data.items()
    }
    
    output = model(input_data)
    # Backbone returns a dictionary or tensor depending on version/config
    if isinstance(output, dict):
        output = output["sequence_output"]
        
    print(f"Process {rank}: Output shape: {output.shape}")
    print(f"Process {rank}: Output is DTensor: {isinstance(output, DTensor)}")

    # Cleanup
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2
    port = random.randint(20000, 30000)
    print(f"Launching {world_size} processes on port {port}...")
    mp.spawn(worker, args=(world_size, port), nprocs=world_size, join=True)
