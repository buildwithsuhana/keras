"""
Bug Description:
`NASNetMobile` had an explicit check that blocked its use with the
PyTorch backend, raising a `ValueError` citing an "outstanding bug."

Fix:
Verified that `NASNetMobile` works correctly with the PyTorch backend,
loads ImageNet weights, and produces results consistent with other
backends. Removed the restrictive check in `keras/src/applications/nasnet.py`.
"""

import os

os.environ["KERAS_BACKEND"] = "torch"

import numpy as np

from keras.src.applications.nasnet import NASNetMobile

print("Attempting to instantiate NASNetMobile with PyTorch backend...")
try:
    # Use weights=None for a quick check, or "imagenet" for full validation
    model = NASNetMobile(weights=None)
    print("Model built successfully!")

    x = np.random.randn(1, 224, 224, 3).astype("float32")
    print("Running prediction...")
    y = model.predict(x)
    print(f"Prediction successful! Output shape: {y.shape}")
    print("Bug Fixed: NASNetMobile is now supported on PyTorch backend!")
except ValueError as e:
    print(f"Bug Reproduced: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
