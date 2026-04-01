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
    if port is not None:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank)

    keras.config.set_floatx("float32")
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
    # Shard the feed-forward layers and attention projections
    layout_map = LayoutMap(mesh)
    # layout_map[".*token_embedding/embeddings"] = (None, "model")
    layout_map[".*query/kernel"] = (None, "model")
    layout_map[".*key/kernel"] = (None, "model")
    layout_map[".*value/kernel"] = (None, "model")
    layout_map[".*attention_output/kernel"] = ("model", None)
    layout_map[".*intermediate_output/kernel"] = (None, "model")
    layout_map[".*layer_output/kernel"] = ("model", None)

    # 5. Create ModelParallel Distribution
    distribution = ModelParallel(
        layout_map=layout_map, 
        batch_dim_name="batch",
        auto_shard_dataset=False
    )

    # 6. Load Model within Distribution Scope
    print(f"Process {rank}: Loading OPT-125M...")
    with distribution.scope():
        # Loading the backbone
        model = keras_hub.models.OPTBackbone.from_preset("opt_125m_en")
    
    # Verify all weights
    if rank == 0:
        for v in model.weights:
            print(f"Weight: {v.path}, sharded: {getattr(v, '_layout', None) is not None}")
    
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

    # Run Forward Pass
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

    # Debug: Check embedding output manually
    with distribution.scope():
        # Input ids are already sharded by distribute_data_input
        token_emb = model.embeddings.token_embedding(input_data["token_ids"])
        print(f"Process {rank}: token_embedding output type: {type(token_emb)}")
        
        pos_emb = model.embeddings.position_embedding(token_emb)
        print(f"Process {rank}: position_embedding output type: {type(pos_emb)}")
        
        # Combined embedding
        combined = model.embeddings(input_data["token_ids"])
        print(f"Process {rank}: model.embeddings() combined output type: {type(combined)}")
        
        # Check dropout or layer norm if they are present in embeddings
        # model.embeddings is usually TokenAndPositionEmbedding
        # It might have a layer_norm or dropout
        if hasattr(model.embeddings, "layer_norm"):
            ln_out = model.embeddings.layer_norm(combined)
            print(f"Process {rank}: embeddings.layer_norm output type: {type(ln_out)}")
        if hasattr(model.embeddings, "dropout"):
            dr_out = model.embeddings.dropout(combined)
            print(f"Process {rank}: embeddings.dropout output type: {type(dr_out)}")

        print(f"Process {rank}: Testing model.fit()...")
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
        
        # Dummy target data (batch_size=1, seq_len=8)
        # Using a simple 1D array as labels, OPTBackbone might need specific shapes
        # If it returns sequence_output (B, S, D), loss might fail without a head.
        # But we can at least see if it executes.
        # Actually, OPTBackbone output is (B, S, D). sparse_categorical_crossentropy 
        # expects (B, S) targets for (B, S, V) outputs.
        # Since we don't have a head, we'll just use a dummy loss or just run fit
        # with a small number of steps.
        
        labels = torch.randint(0, 100, (1, 8))
        
        # We need to distribute labels as well
        labels = torch_dist_lib.distribute_data_input(
            labels, distribution.get_data_layout(labels.shape), distribution.batch_dim_name
        )
        
        # backbone doesn't have a head, so we might need to add one or use a custom loss
        # to avoid shape mismatch. 
        # Let's just try to run it.
        try:
            model.fit(input_data, labels, epochs=1, steps_per_epoch=1)
            print(f"Process {rank}: model.fit() SUCCESS")
        except Exception as e:
            print(f"Process {rank}: model.fit() FAILED with {e}")
            raise e

    # Cleanup
    dist.destroy_process_group()

if __name__ == "__main__":
    if "WORLD_SIZE" in os.environ:
        # Already in a distributed environment (e.g. torchrun)
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        worker(rank, world_size, None)
    else:
        # Local execution without torchrun
        world_size = 2
        port = random.randint(20000, 30000)
        print(f"Launching {world_size} processes on port {port}...")
        mp.spawn(worker, args=(world_size, port), nprocs=world_size, join=True)
