import os
os.environ["KERAS_BACKEND"] = "torch"

import keras
import numpy as np
import torch

def test_in_top_k_3d_values():
    # Predictions: (batch=1, seq=2, classes=3)
    predictions = np.array([
        [
            [0.1, 0.2, 0.7], # max is index 2
            [0.8, 0.1, 0.1]  # max is index 0
        ]
    ]).astype("float32")
    
    # Targets: (batch=1, seq=2)
    # Correct targets (both are max)
    targets = np.array([[2, 0]]).astype("int32")
    
    print("Testing CORRECT targets (expect [True, True])")
    out = keras.ops.in_top_k(targets, predictions, k=1)
    print(f"Output: {out}")
    
    # Incorrect targets: seq 0 target 0 (val 0.1), seq 1 target 2 (val 0.1)
    # Neither is max.
    targets_bad = np.array([[0, 2]]).astype("int32")
    print("\nTesting INCORRECT targets (expect [False, False])")
    out_bad = keras.ops.in_top_k(targets_bad, predictions, k=1)
    print(f"Output: {out_bad}")

    # If the bug is what I think (targets = targets[:, None]), 
    # then for targets_bad=[0, 2], targets[:, None] is [[0, 2]] (1, 2).
    # It gets broadcasted to [[0, 2], [0, 2]] (2, 2) when matching predictions[0] (2, 3).
    # So for predictions[0, 0], it checks indices [0, 2]. Val at index 2 is 0.7, which IS top-1.
    # So it might return True!

if __name__ == "__main__":
    test_in_top_k_3d_values()
