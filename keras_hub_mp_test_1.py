import os
import sys
import logging
# MUST be set before any other imports
os.environ["KERAS_BACKEND"] = "torch"
os.environ["KERAS_DISTRIBUTION_DEBUG"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import keras
import keras_hub
import numpy as np
import torch
from keras.src.distribution import ModelParallel, DeviceMesh, LayoutMap, list_devices, initialize
from keras.src.backend.torch.distribution_lib import set_mp_multi_process_state, prepare_input_for_distribution

# Setup logging
logging.basicConfig(level=logging.INFO)

def log(msg):
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    print(f"TEST | [Rank {rank:02d}] {msg}")

def test_opt_model_parallel():
    # 1. Initialize
    initialize()
    devices = list_devices("gpu")
    log(f"Devices detected: {devices}")
    
    mesh = DeviceMesh(shape=(1, len(devices)), axis_names=["batch", "model"], devices=devices)
    
    layout_map = LayoutMap(mesh)
    layout_map["embeddings.token_embedding.embeddings"] = (None, "model")
    layout_map["embeddings.position_embedding.embeddings"] = (None, None)
    layout_map["transformer_layer_.*.attention.*.kernel"] = (None, "model")
    layout_map["transformer_layer_.*.feedforward.*.kernel"] = (None, "model")
    layout_map[".*layer_norm.*"] = ()

    mp = ModelParallel(layout_map=layout_map, batch_dim_name="batch", auto_shard_dataset=False)
    set_mp_multi_process_state(True)

    # 2. Build Model
    with mp.scope():
        log("Creating OPT model...")
        model = keras_hub.models.OPTBackbone(
            vocabulary_size=50265, num_layers=2, num_heads=2,
            hidden_dim=128, intermediate_dim=256, max_sequence_length=32
        )
        
        log("Building model variables...")
        model.build({"token_ids": (4, 16), "padding_mask": (4, 16)})
        
        loss_fn = keras.losses.MeanSquaredError()
        optimizer = keras.optimizers.Adam(1e-3)
        optimizer.build(model.trainable_variables)

    # 3. Prepare Data (Promote to DTensors)
    x_raw = {
        "token_ids": np.random.randint(0, 50265, size=(4, 16), dtype="int32"),
        "padding_mask": np.ones((4, 16), dtype="int32"),
    }
    y_raw = np.random.random((4, 16, 128)).astype("float32")

    log("Distributing input data...")
    with mp.scope():
        # Uses the fixed _convert_structure logic to create Replicated DTensors
        x_dist = prepare_input_for_distribution(x_raw)
        y_dist = prepare_input_for_distribution(y_raw)

    # 4. Custom Training Loop (Bypasses problematic fit() logic)
    log("Starting Training Step...")
    
    # We execute eagerly to avoid the Dynamo IndexError
    with mp.scope():
        # Forward pass
        output = model(x_dist, training=True)
        loss = loss_fn(y_dist, output)
        
        log(f"Loss computed: {float(loss):.6f}")
        
        # Backward pass
        # model.trainable_variables are DTensors, so grads will be DTensors
        grads = torch.autograd.grad(loss, model.trainable_variables)
        
        # Apply gradients
        optimizer.apply(grads, model.trainable_variables)
        
    log("✓ ModelParallel test PASSED")

if __name__ == "__main__":
    try:
        test_opt_model_parallel()
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)