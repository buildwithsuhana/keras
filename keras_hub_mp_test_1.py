import os
# Set backend to torch
os.environ["KERAS_BACKEND"] = "torch"
os.environ["KERAS_DISTRIBUTION_DEBUG"] = "1"

import keras
import keras_hub
import numpy as np
import torch
from keras.src.distribution import ModelParallel, DeviceMesh, LayoutMap, list_devices, initialize
from keras.src.backend.torch.distribution_lib import set_mp_multi_process_state

def log(msg):
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    print(f"TEST | [Rank {rank:02d}] {msg}")

def test_opt_model_parallel():
    initialize()
    devices = list_devices("gpu")
    mesh = DeviceMesh(shape=(1, len(devices)), axis_names=["batch", "model"], devices=devices)
    
    layout_map = LayoutMap(mesh)
    # Token embeddings sharded, Position embeddings replicated DTensor
    layout_map["embeddings.token_embedding.embeddings"] = (None, "model")
    layout_map["embeddings.position_embedding.embeddings"] = (None, None)
    layout_map["transformer_layer_.*.attention.*.kernel"] = (None, "model")
    layout_map["transformer_layer_.*.feedforward.*.kernel"] = (None, "model")
    layout_map[".*layer_norm.*"] = ()

    mp = ModelParallel(layout_map=layout_map, batch_dim_name="batch", auto_shard_dataset=False)
    
    # Sync backend state
    set_mp_multi_process_state(True)

    with mp.scope():
        log("Creating OPT model...")
        model = keras_hub.models.OPTBackbone(
            vocabulary_size=50265, num_layers=2, num_heads=2,
            hidden_dim=128, intermediate_dim=256, max_sequence_length=32
        )
        
        # CRITICAL: Build inside scope so weights are created as DTensors
        log("Building model...")
        model.build({"token_ids": (4, 16), "padding_mask": (4, 16)})
        
        model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")

    # Raw Numpy inputs - Backend _convert_structure will handle DTensor wrapping
    x = {
        "token_ids": np.random.randint(0, 50265, size=(4, 16), dtype="int32"),
        "padding_mask": np.ones((4, 16), dtype="int32"),
    }
    y = np.random.random((4, 16, 128)).astype("float32")

    log("Starting fit...")
    with mp.scope():
        model.fit(x, y, epochs=1, verbose=0)
    
    log("✓ ModelParallel test PASSED")

if __name__ == "__main__":
    test_opt_model_parallel()