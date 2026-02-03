#!/usr/bin/env python3
import os

# MUST be set before importing Keras
os.environ["KERAS_BACKEND"] = "torch"

import torch
import torch.distributed as dist
import keras
import keras_hub
import numpy as np
import requests
from keras.distribution import ModelParallel, DeviceMesh, LayoutMap, list_devices, initialize

def get_tiny_shakespeare():
    """Simple helper to get text data."""
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    return requests.get(url).text[:50000]

def run_hybrid_parallel():
    # 1. Initialize distributed environment
    # This automatically detects world_size and rank if using torchrun
    initialize()
    
    devices = list_devices("gpu")
    num_devices = len(devices)
    
    if num_devices < 2:
        print(f"Error: Found only {num_devices} GPU(s). Need at least 2 for Hybrid Parallelism.")
        return

    # 2. Define 2D Device Mesh for (Data Parallel, Model Parallel)
    # Example: With 4 GPUs, shape=(2, 2) means 2-way DP and 2-way MP.
    # With 2 GPUs, we can do (1, 2) for pure MP or (2, 1) for pure DP.
    # To demonstrate both, we need at least 4 GPUs, or we use (1, 2) on 2 GPUs.
    dp_size = 2 if num_devices >= 4 else 1
    mp_size = num_devices // dp_size
    
    mesh = DeviceMesh(
        shape=(dp_size, mp_size),
        axis_names=["data", "model"],
        devices=devices
    )
    print(f"Created Mesh: {dp_size}-way Data Parallel, {mp_size}-way Model Parallel")

    # 3. Create LayoutMap for Sharding
    # We shard the "model" axis and replicate on the "data" axis (None)
    layout_map = LayoutMap(mesh)
    layout_map["token_embedding/embeddings"] = (None, "model")
    layout_map[".*attention.*query.*kernel"] = (None, "model")
    layout_map[".*attention.*key.*kernel"] = (None, "model")
    layout_map[".*attention.*value.*kernel"] = (None, "model")
    layout_map[".*attention.*output_dense.*kernel"] = ("model", None) # Row-parallel
    
    # 4. Create the Combined Strategy
    # 'batch_dim_name' tells Keras which mesh axis to use for Data Parallelism
    strategy = ModelParallel(
        layout_map=layout_map,
        batch_dim_name="data", 
        auto_shard_dataset=True
    )

    # 5. Load Model within Strategy Scope
    with strategy.scope():
        model = keras_hub.models.OPTCausalLM.from_preset("opt_125m_en")
        model.compile(
            optimizer=keras.optimizers.Adam(1e-5),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        )

    # 6. Prepare Data (Tiny Shakespeare)
    text = get_tiny_shakespeare()
    # Create simple samples
    samples = [text[i:i+128] for i in range(0, 1024, 128)]
    
    # 7. Train
    print("Starting Hybrid Training...")
    model.fit(samples, epochs=1, batch_size=dp_size * 2)
    
    # 8. Inference Verification
    if dist.get_rank() == 0:
        print("\nTesting Generation:")
        print(model.generate("ROMEO: ", max_length=40))

if __name__ == "__main__":
    run_hybrid_parallel()