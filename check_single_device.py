import os
import numpy as np
import json
import keras
import keras_hub
import sys

def check(backend):
    print(f"\n--- Checking {backend.upper()} ---")
    os.environ["KERAS_BACKEND"] = backend
    if backend == "torch":
        import torch
        if hasattr(torch, "set_default_device"): torch.set_default_device("cpu")

    keras.utils.set_random_seed(42)
    model = keras_hub.models.OPTBackbone.from_preset("opt_125m_en")
    model.load_weights("initial_weights.weights.h5")
    
    for layer in model._flatten_layers():
        for attr in ["dropout", "dropout_rate", "hidden_dropout_rate", "attention_dropout_rate"]:
            if hasattr(layer, attr):
                try: setattr(layer, attr, 0.0)
                except: pass
    
    data = np.load("test_data.npz")
    x_tokens = data["x_tokens"][:2]
    x_mask = data["x_mask"][:2]
    y = data["y"][:2]
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    )
    
    print("Checking sample 0...")
    h0 = model.evaluate({"token_ids": x_tokens[0:1], "padding_mask": x_mask[0:1]}, y[0:1], verbose=0)
    print(f"Sample 0 Loss: {h0:.8f}")
    
    print("Checking sample 1...")
    h1 = model.evaluate({"token_ids": x_tokens[1:2], "padding_mask": x_mask[1:2]}, y[1:2], verbose=0)
    print(f"Sample 1 Loss: {h1:.8f}")
    
    print(f"Mean Loss: {(h0 + h1) / 2:.8f}")

if __name__ == "__main__":
    check("jax")
    # Need to run torch in a separate process because Keras backend cannot be changed in the same process
    import subprocess
    subprocess.run([sys.executable, __file__, "torch"])

if len(sys.argv) > 1 and sys.argv[1] == "torch":
    check("torch")
