import os
# Must be set before any other imports
os.environ["KERAS_BACKEND"] = "torch"

import torch
import torch.distributed as dist
import keras
import keras_hub
import numpy as np
import requests
from keras.src.distribution import ModelParallel, DeviceMesh, LayoutMap, list_devices, initialize

def get_data():
    """Fetch and preprocess Tiny Shakespeare."""
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    text = requests.get(url).text[:50000] 
    # Create simple 128-token sequences for training
    # In a real scenario, use a proper KerasHub preprocessor/tokenizer
    return [text[i : i + 128] for i in range(0, len(text) - 128, 128)]

def run_training():
    # 1. Initialize Distributed Backend
    initialize()
    devices = list_devices("gpu")
    num_devices = len(devices)
    
    if num_devices < 2:
        print(f"Detected {num_devices} GPU(s). Need at least 2 for MP+DP demo.")
        return

    # 2. Define 2D Device Mesh (DP x MP)
    # On 2 GPUs: (2, 1) = Pure DP; (1, 2) = Pure MP
    # On 4 GPUs: (2, 2) = Hybrid (2-way Data Parallel, 2-way Model Parallel)
    dp_dim = 2 if num_devices >= 4 else 1
    mp_dim = num_devices // dp_dim
    
    mesh = DeviceMesh(
        shape=(dp_dim, mp_dim),
        axis_names=["data", "model"],
        devices=devices
    )

    # 3. Create Sharding Layout
    # Shard weights along the "model" axis and replicate across "data" axis
    layout_map = LayoutMap(mesh)
    layout_map["token_embedding/embeddings"] = (None, "model")
    layout_map[".*attention.*query.*kernel"] = (None, "model")
    layout_map[".*attention.*key.*kernel"] = (None, "model")
    layout_map[".*attention.*value.*kernel"] = (None, "model")
    layout_map[".*attention.*output_dense.*kernel"] = ("model", None) # Row-parallel
    
    # 4. Initialize Strategy
    # batch_dim_name="data" tells Keras to split the dataset across the 'data' axis
    strategy = ModelParallel(
        layout_map=layout_map,
        batch_dim_name="data"
    )

    # 5. Compile & Build Model within Strategy Scope
    with strategy.scope():
        # Load the OPT-125m causal language model
        model = keras_hub.models.OPTCausalLM.from_preset("opt_125m_en")
        
        # Standard Keras Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=5e-5),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            weighted_metrics=["accuracy"]
        )

    # 6. Load Data
    train_data = get_data()

    # 7. Start Training with model.fit()
    print(f"Rank {dist.get_rank()}: Starting Fit...")
    model.fit(
        train_data, 
        epochs=1, 
        batch_size=dp_dim * 2, # Global batch size distributed across DP nodes
        verbose=1 if dist.get_rank() == 0 else 0
    )

    # 8. Post-training Inference
    if dist.get_rank() == 0:
        print("\nVerification Generation:")
        print(model.generate("ROMEO: ", max_length=30))

if __name__ == "__main__":
    run_training()