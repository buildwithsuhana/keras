import os

os.environ["KERAS_BACKEND"] = "torch"

import keras

try:
    v = keras.Variable([1, 2, 3])
    print("Variable created successfully")
    print(v)
except Exception as e:
    print(f"Error creating Variable: {e}")

try:
    v = keras.Variable([1, 2, 3], trainable=True)
    print("Variable (trainable=True) created successfully")
    print(v)
except Exception as e:
    print(f"Error creating Variable (trainable=True): {e}")

try:
    v = keras.Variable([1.0, 2.0, 3.0], trainable=True)
    print("Float Variable created successfully")
    print(v)
except Exception as e:
    print(f"Error creating Float Variable: {e}")
