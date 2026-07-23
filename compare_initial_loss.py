import os
import subprocess
import sys

import numpy as np


def run_backend(backend):
    os.environ["KERAS_BACKEND"] = backend
    if backend == "jax":
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"

    import keras_hub

    import keras

    keras.utils.set_random_seed(42)

    model = keras_hub.models.OPTBackbone.from_preset("opt_125m_en", dropout=0.0)
    model.compile(optimizer="adam", loss="mse")

    np.random.seed(42)
    x = {
        "token_ids": np.random.randint(0, 50272, (1, 32)).astype("int32"),
        "padding_mask": np.ones((1, 32), dtype="int32"),
    }
    y = np.random.normal(size=(1, 32, 768)).astype("float32")

    loss = model.evaluate(x, y, verbose=0)
    print(f"{backend} initial loss: {loss:.12f}")

    # Also check first few weights to ensure they are the same
    weights = model.get_weights()
    print(f"{backend} first weight mean: {np.mean(weights[0]):.12f}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_backend(sys.argv[1])
    else:
        print("Running JAX...")
        subprocess.run([sys.executable, __file__, "jax"])
        print("Running Torch...")
        subprocess.run([sys.executable, __file__, "torch"])
