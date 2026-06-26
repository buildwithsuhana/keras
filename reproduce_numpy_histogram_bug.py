"""
Bug Description:
The NumPy backend's `compute_output_spec` used `np.empty` to create
dummy tensors for shape/dtype inference. `np.empty` doesn't initialize
memory, so it can contain NaNs. Operations like `keras.ops.histogram`
validate their input range and crash if they encounter NaNs during this
dry-run phase.

Fix:
Replaced `np.empty` with `np.zeros` in `keras/src/backend/numpy/core.py`
to ensure dummy tensors have predictable, finite values.
"""

import os

os.environ["KERAS_BACKEND"] = "numpy"

import numpy as np

import keras
from keras.src import ops
from keras.src.ops import numpy as knp


class HistogramLayer(keras.layers.Layer):
    def call(self, x):
        shape = ops.shape(x)
        # Flatten, because the op does not work with >1-dim inputs.
        x = ops.reshape(x, (shape[0] * shape[1],))
        return knp.histogram(x, bins=5)


print("Attempting to build a model with HistogramLayer on NumPy backend...")
inputs = keras.Input(shape=(8,))
try:
    # This triggers compute_output_spec, which would previously fail
    # if np.empty returned NaNs
    outputs = HistogramLayer()(inputs)
    model = keras.Model(inputs, outputs)
    model.compile()

    print("Dry-run successful. Running model.predict...")
    model.predict(np.random.randn(1, 8))
    print("Bug Fixed: Model predict successful on NumPy backend!")
except ValueError as e:
    print(f"Bug Reproduced: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
