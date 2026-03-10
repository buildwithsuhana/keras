import os
import keras
import numpy as np
import keras_hub

# Using default float32
keras.config.set_floatx("float32")

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
    model = keras_hub.models.OPTBackbone(**config)
    
    # Build and save
    model({"token_ids": np.zeros((1, 32), dtype="int32"), "padding_mask": np.ones((1, 32), dtype="int32")})
    model.save_weights("initial_weights.weights.h5")
    
    # Generate 256 samples
    np.random.seed(42)
    token_ids = np.random.randint(0, 1000, (256, 32)).astype("int32")
    padding_mask = np.ones((256, 32), dtype="int32")
    y = np.random.randn(256, 32, 64).astype("float32")
    
    np.savez("data.npz", token_ids=token_ids, padding_mask=padding_mask, y=y)
    print("Initial weights and data (float32) saved.")

if __name__ == "__main__":
    generate_initial_state()
