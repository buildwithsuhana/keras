
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import numpy as np
from keras.src import layers
from keras.src import backend

def test_cross_output_dtype():
    input_1, input_2 = np.array([1]), np.array([1])

    print("Testing int64 default...")
    layer = layers.HashedCrossing(num_bins=2)
    output = layer((input_1, input_2))
    output_dtype = backend.standardize_dtype(output.dtype)
    print(f"Output dtype: {output_dtype}")
    assert output_dtype == "int64"

    print("Testing int32 override...")
    layer = layers.HashedCrossing(num_bins=2, dtype="int32")
    output = layer((input_1, input_2))
    output_dtype = backend.standardize_dtype(output.dtype)
    print(f"Output dtype: {output_dtype}")
    assert output_dtype == "int32"

    print("Testing one_hot float32 default...")
    layer = layers.HashedCrossing(num_bins=2, output_mode="one_hot")
    output = layer((input_1, input_2))
    output_dtype = backend.standardize_dtype(output.dtype)
    print(f"Output dtype: {output_dtype}")
    assert output_dtype == "float32"

    print("Testing one_hot float64 override...")
    layer = layers.HashedCrossing(num_bins=2, output_mode="one_hot", dtype="float64")
    output = layer((input_1, input_2))
    output_dtype = backend.standardize_dtype(output.dtype)
    print(f"Output dtype: {output_dtype}")
    assert output_dtype == "float64"

if __name__ == "__main__":
    try:
        test_cross_output_dtype()
        print("All tests passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
