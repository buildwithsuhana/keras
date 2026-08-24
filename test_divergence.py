import os
import numpy as np
import keras
from keras import ops

def test_divergence():
    backend = keras.backend.backend()
    print(f"Testing backend: {backend}")
    
    # Simple model
    model = keras.Sequential([
        keras.layers.Dense(10, input_shape=(5,), use_bias=False, kernel_initializer="ones")
    ])
    
    # Fixed input
    x = ops.ones((2, 5))
    y = ops.ones((2, 10))
    
    # Fixed weights
    model.layers[0].kernel.assign(ops.ones((5, 10)) * 0.1)
    
    # Initial Loss
    logits = model(x)
    loss_fn = keras.losses.MeanSquaredError()
    initial_loss = loss_fn(y, logits)
    print(f"INITIAL_LOSS: {float(initial_loss):.12f}")
    
    # One training step
    opt = keras.optimizers.SGD(1e-2)
    model.compile(optimizer=opt, loss=loss_fn)
    model.train_on_batch(x, y)
    
    # Loss after one step
    logits_after = model(x)
    final_loss = loss_fn(y, logits_after)
    print(f"FINAL_LOSS_AFTER_FIT: {float(final_loss):.12f}")

if __name__ == "__main__":
    test_divergence()
