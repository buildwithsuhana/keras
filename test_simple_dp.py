#!/usr/bin/env python3
"""Simple test for the distributed training fix."""

import os
# MUST be set before any other imports
os.environ["KERAS_BACKEND"] = "torch"
os.environ["KERAS_DISTRIBUTION_DEBUG"] = "1"

import keras
from keras import layers
from keras.src.distribution import DataParallel, list_devices
import numpy as np

print("=" * 70)
print("SIMPLE DATA PARALLEL TEST")
print("=" * 70)

# Get devices
devices = list_devices("gpu")
if not devices:
    devices = ["cpu:0"]

print(f"Devices: {devices}")

# Create DataParallel distribution
dp = DataParallel(devices=devices, auto_shard_dataset=False)
print(f"DataParallel created: mesh_shape={dp.device_mesh.shape}")

# Create model inside the distribution scope
print("\nCreating model inside distribution scope...")
with dp.scope():
    model = keras.Sequential([
        layers.Dense(128, activation="relu", input_shape=(64,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(10)
    ])
    
    print(f"Model created with {model.count_params()} parameters")
    
    # Check the parameters
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'kernel'):
            kernel = layer.kernel
            print(f"  Layer {i}: {layer.name}")
            print(f"    - kernel._layout: {kernel._layout}")
            if hasattr(kernel, '_value'):
                print(f"    - kernel._value type: {type(kernel._value)}")
                print(f"    - kernel._value.requires_grad: {kernel._value.requires_grad}")
    
    model.compile(optimizer="adam", loss="mse")

# Create training data
batch_size = 32
x = np.random.random((batch_size, 64)).astype("float32")
y = np.random.random((batch_size, 10)).astype("float32")

# Training loop
print("\nTraining for 3 epochs...")
for epoch in range(3):
    with dp.scope():
        history = model.fit(x, y, epochs=1, verbose=0)
        loss = history.history['loss'][0]
    print(f"  Epoch {epoch+1}/3: loss={loss:.6f}")

print("\n" + "=" * 70)
print("TEST PASSED!")
print("=" * 70)

