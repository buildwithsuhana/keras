import os
os.environ["KERAS_BACKEND"] = "torch"
os.environ["KERAS_DISTRIBUTION_DEBUG"] = "1"

import keras
import keras_hub
import numpy as np
import torch
from keras.src.distribution import ModelParallel, DeviceMesh, LayoutMap, list_devices, initialize
from keras.src.backend.torch.distribution_lib import set_mp_multi_process_state, prepare_input_for_distribution

def log(msg):
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    print(f"TEST | [Rank {rank:02d}] {msg}")

def test_opt_model_parallel():
    initialize()
    devices = list_devices("gpu")
    mesh = DeviceMesh(shape=(1, len(devices)), axis_names=["batch", "model"], devices=devices)
    
    layout_map = LayoutMap(mesh)
    layout_map["embeddings.token_embedding.embeddings"] = (None, "model")
    layout_map["embeddings.position_embedding.embeddings"] = (None, None)
    layout_map["transformer_layer_.*.attention.*.kernel"] = (None, "model")
    layout_map["transformer_layer_.*.feedforward.*.kernel"] = (None, "model")
    layout_map[".*layer_norm.*"] = ()

    mp = ModelParallel(layout_map=layout_map, batch_dim_name="batch", auto_shard_dataset=False)
    set_mp_multi_process_state(True)

    with mp.scope():
        log("Creating OPT model...")
        model = keras_hub.models.OPTBackbone(
            vocabulary_size=50265, num_layers=2, num_heads=2,
            hidden_dim=128, intermediate_dim=256, max_sequence_length=32
        )
        model.build({"token_ids": (4, 16), "padding_mask": (4, 16)})
        
        loss_fn = keras.losses.MeanSquaredError()
        optimizer = keras.optimizers.Adam(1e-3)
        # We manually build the optimizer variables inside the scope
        optimizer.build(model.trainable_variables)

    # 1. Prepare Data as DTensors
    x_raw = {
        "token_ids": np.random.randint(0, 50265, size=(4, 16), dtype="int32"),
        "padding_mask": np.ones((4, 16), dtype="int32"),
    }
    y_raw = np.random.random((4, 16, 128)).astype("float32")

    log("Distributing data...")
    with mp.scope():
        x_dist = prepare_input_for_distribution(x_raw)
        y_dist = prepare_input_for_distribution(y_raw)

    # 2. Custom Training Step (Bypasses DataAdapter/DataLoader stripping)
    log("Starting Custom Training Step...")
    import contextlib
    @torch.compile(backend="eager") # Optional: helps in distributed debugging
    def train_step(data, target):
        with torch.amp.autocast('cuda') if torch.cuda.is_available() else contextlib.nullcontext():
            with mp.scope():
                output = model(data, training=True)
                loss = loss_fn(target, output)
        
        # In PyTorch Keras, we use the optimizer to handle sharded gradients
        gradients = torch.autograd.grad(loss, model.trainable_variables)
        optimizer.apply(gradients, model.trainable_variables)
        return loss

    # Run one step
    try:
        loss_val = train_step(x_dist, y_dist)
        log(f"✓ Training step successful. Loss: {float(loss_val):.6f}")
        log("✓ ModelParallel test PASSED")
    except Exception as e:
        log(f"FAILED during train_step: {e}")
        raise e

if __name__ == "__main__":
    test_opt_model_parallel()