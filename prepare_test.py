import os
import numpy as np
import keras

# Force CPU for consistent weight generation
os.environ["KERAS_BACKEND"] = "jax"
import keras_hub

def prepare():
    print("Generating data...")
    np.random.seed(42)
    x_tokens = np.random.randint(0, 50272, (4, 32)).astype("int32")
    x_mask = np.ones((4, 32), dtype="int32")
    # hidden_dim is 768
    y = np.random.randint(0, 768, (4, 32)).astype("int32")
    print(f"y max: {np.max(y)}, y shape: {y.shape}")
    np.savez("test_data.npz", x_tokens=x_tokens, x_mask=x_mask, y=y)

    print("Generating initial weights...")
    model = keras_hub.models.OPTBackbone.from_preset("opt_125m_en")
    model.save_weights("initial_weights.weights.h5")
    print("Preparation complete.")

if __name__ == "__main__":
    prepare()
