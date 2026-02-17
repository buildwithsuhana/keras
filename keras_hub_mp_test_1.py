import os
# MUST be set before any other imports
os.environ["KERAS_BACKEND"] = "torch"
os.environ["KERAS_DISTRIBUTION_DEBUG"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import keras
import keras_hub
import numpy as np
import torch
import sys
from keras.src.distribution import ModelParallel, DeviceMesh, LayoutMap, list_devices, initialize
from keras.src.backend.torch.distribution_lib import set_mp_multi_process_state, prepare_input_for_distribution

def log(msg):
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    print(f"TEST | [Rank {rank:02d}] {msg}")

def test_opt_model_parallel():
    # Initialize the distributed system
    initialize()
    
    devices = list_devices("gpu")
    if len(devices) < 2:
        print("Need at least 2 GPUs for ModelParallel test.")
        return

    # Create a 1D logical mesh for multi-process isolation
    mesh = DeviceMesh(shape=(1, len(devices)), axis_names=["batch", "model"], devices=devices)
    
    layout_map = LayoutMap(mesh)
    # Sharding Configuration
    layout_map["embeddings.token_embedding.embeddings"] = (None, "model")
    layout_map["embeddings.position_embedding.embeddings"] = (None, None) # Replicated DTensor
    layout_map["transformer_layer_.*.attention.*.kernel"] = (None, "model")
    layout_map["transformer_layer_.*.feedforward.*.kernel"] = (None, "model")
    layout_map[".*layer_norm.*"] = ()

    mp = ModelParallel(layout_map=layout_map, batch_dim_name="batch", auto_shard_dataset=False)
    
    # Sync backend state to handle multi-process DTensor wrapping
    set_mp_multi_process_state(True)

    with mp.scope():
        log("Creating OPT model...")
        model = keras_hub.models.OPTBackbone(
            vocabulary_size=50265, num_layers=2, num_heads=2,
            hidden_dim=128, intermediate_dim=256, max_sequence_length=32
        )
        
        # CRITICAL: Build inside scope so weights are created as sharded DTensors
        log("Building model...")
        model.build({"token_ids": (4, 16), "padding_mask": (4, 16)})
        
        model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")

    # Generate raw data
    x_raw = {
        "token_ids": np.random.randint(0, 50265, size=(4, 16), dtype="int32"),
        "padding_mask": np.ones((4, 16), dtype="int32"),
    }
    y_raw = np.random.random((4, 16, 128)).astype("float32")

    # CRITICAL: Explicitly distribute data into DTensors to prevent mixed-type errors
    log("Explicitly distributing input data into DTensors...")
    with mp.scope():
        x_dist = prepare_input_for_distribution(x_raw)
        y_dist = prepare_input_for_distribution(y_raw)
    
    log(f"Verification: token_ids is now {type(x_dist['token_ids'])}")

    log("Starting fit...")
    with mp.scope():
        history = model.fit(x_dist, y_dist, epochs=1, verbose=0)
        log(f"  Final Loss: {history.history['loss'][0]:.6f}")
    
    log("✓ ModelParallel test PASSED")

if __name__ == "__main__":
    try:
        test_opt_model_parallel()
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)