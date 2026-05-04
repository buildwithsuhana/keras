import os
os.environ["KERAS_BACKEND"] = "torch"
import keras
print("Torch devices:", keras.distribution.list_devices())

os.environ["KERAS_BACKEND"] = "jax"
import jax
print("JAX devices:", jax.devices())
