
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import numpy as np
import tensorflow as tf

def test_multi_hot_3d():
    # Shape: (batch=2, seq=3, indices=2)
    # We expect output shape: (2, 3, num_classes=5)
    x = np.array([
        [[0, 1], [2, 3], [0, 4]],
        [[1, 2], [3, 4], [0, 2]]
    ])
    num_classes = 5
    
    print(f"Input shape: {x.shape}")
    try:
        y = keras.ops.multi_hot(x, num_classes=num_classes)
        print(f"Output shape: {y.shape}")
        
        # Expected shape (2, 3, 5)
        # If bug exists, reduction_axis will be 1, so it reduces over 'seq' (dim 1)
        # resulting in (2, 2, 5) if one_hot was (2, 3, 2, 5)
        # Wait, if one_hot is (2, 3, 2, 5) and we reduce dim 1, result is (2, 2, 5).
        
        expected_shape = (2, 3, 5)
        if y.shape != expected_shape:
            print(f"BUG DETECTED! Expected shape {expected_shape}, but got {y.shape}")
        else:
            print("Shape is correct. Checking values...")
            # Check a value: x[0, 0] = [0, 1] -> y[0, 0] should be [1, 1, 0, 0, 0]
            expected_val = [1, 1, 0, 0, 0]
            if not np.array_equal(y[0, 0], expected_val):
                print(f"VALUE BUG! Expected {expected_val} at y[0,0], but got {y[0,0]}")
            else:
                print("Values seem correct for y[0,0].")

    except Exception as e:
        print(f"Failed with error: {e}")

if __name__ == "__main__":
    test_multi_hot_3d()
