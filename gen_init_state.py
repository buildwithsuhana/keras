import os
import keras
import numpy as np
import keras_hub

# Force float64 for verification
keras.config.set_floatx("float64")

def generate_initial_state():
    config = {
        "vocabulary_size": 1000,
        "num_layers": 2,
        "num_heads": 2,
        "hidden_dim": 64,
        "intermediate_dim": 128,
        "max_sequence_length": 32,
        "dropout": 0.0,
    }
    
    os.environ["KERAS_BACKEND"] = "jax"
    os.environ["JAX_ENABLE_X64"] = "True"

    model = keras_hub.models.OPTBackbone(**config)
    # Initialize weights
    model({"token_ids": np.zeros((1, 32), dtype="int32"), "padding_mask": np.ones((1, 32), dtype="int32")})
    model.save_weights("initial_weights.weights.h5")
    
    np.random.seed(42)
    token_ids = np.random.randint(0, 1000, (256, 32)).astype("int32")
    padding_mask = np.ones((256, 32), dtype="int32")
    # Use random targets for MSE
    y = np.random.randn(256, 32, 64).astype("float64")
    
    np.savez("data.npz", token_ids=token_ids, padding_mask=padding_mask, y=y)
    print("Initial weights and data (float64) saved.")

if __name__ == "__main__":
    generate_initial_state()
