import os
import sys
import numpy as np

def run_jax():
    os.environ["KERAS_BACKEND"] = "jax"
    import keras
    import keras_hub
    keras.utils.set_random_seed(42)
    model = keras_hub.models.OPTBackbone.from_preset("opt_125m_en", dropout=0.0)
    
    x = {
        "token_ids": np.ones((1, 32), dtype="int32"),
        "padding_mask": np.ones((1, 32), dtype="int32"),
    }
    out = model.predict(x, verbose=0)
    np.save("jax_out.npy", out)

def run_torch():
    os.environ["KERAS_BACKEND"] = "torch"
    import keras
    import keras_hub
    keras.utils.set_random_seed(42)
    model = keras_hub.models.OPTBackbone.from_preset("opt_125m_en", dropout=0.0)
    
    x = {
        "token_ids": np.ones((1, 32), dtype="int32"),
        "padding_mask": np.ones((1, 32), dtype="int32"),
    }
    out = model.predict(x, verbose=0)
    np.save("torch_out.npy", out)

if __name__ == "__main__":
    if sys.argv[1] == "jax":
        run_jax()
    elif sys.argv[1] == "torch":
        run_torch()
    elif sys.argv[1] == "compare":
        jax_out = np.load("jax_out.npy")
        torch_out = np.load("torch_out.npy")
        print(f"JAX out mean: {np.mean(jax_out)}")
        print(f"Torch out mean: {np.mean(torch_out)}")
        print(f"Max diff: {np.max(np.abs(jax_out - torch_out))}")
        print(f"JAX out samples: {jax_out.flatten()[:5]}")
        print(f"Torch out samples: {torch_out.flatten()[:5]}")
