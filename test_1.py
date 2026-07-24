import os

# 1. Configuration: Enable JAX backend and NNX integration
os.environ["KERAS_BACKEND"] = "jax"
os.environ["KERAS_NNX_ENABLED"] = "true"

import jax.numpy as jnp

import keras

try:
    from keras.src.utils.tunix_utils import TunixKerasAdapter

    print("✅ Success: TunixKerasAdapter imported!")
except ImportError as e:
    print("❌ Failure: Failed to import TunixKerasAdapter.")
    print("Reason:", e)
    print("This confirms the script fails without the integration changes.")
    exit(1)
print("Attempting to import TunixKerasAdapter...")

# 2. Create a Dummy Keras Model
inputs = keras.Input(shape=(10,), dtype="int32", name="input_tokens")
x = keras.layers.Embedding(input_dim=1000, output_dim=64)(inputs)
x = keras.layers.GlobalAveragePooling1D()(x)
outputs = keras.layers.Dense(10, activation="softmax")(x)
model = keras.Model(inputs=inputs, outputs=outputs)

print("Keras Model created successfully.")
print("\nWrapping Keras model with TunixKerasAdapter...")
# 4. Wrap with Adapter
adapter = TunixKerasAdapter(model, embed_dim=64, num_layers=1)
print("✅ Success: Model wrapped!")

# 3. Create Dummy Inputs matching Tunix Pipeline Expectations
input_tokens = jnp.ones((2, 10), dtype="int32")
attention_mask = jnp.ones((2, 10), dtype="bool")
positions = jnp.arange(10)[None, :]
cache = None  # No cache for this simple example

print("\nAttempting to call Keras model directly with Tunix signature...")
print("Signature: model(input_tokens, positions, cache, attention_mask)")

try:
    # 💥 THIS STEP WILL FAIL
    logits = adapter(input_tokens, positions, cache, attention_mask)

    print("✅ Success? That shouldn't have happened!")
    print(f"Return type: {type(logits)}")

except TypeError as e:
    print("\n❌ Failure: TypeError Raised as expected!")
    print(f"Error Message: {e}")
    print("\n💡 Why did this happen?")
    print("Standard Keras models: model(inputs, training=None, mask=None)")
    print(
        "KKeras tries to match 'positions' to 'training' and 'cache' to 'mask'!"
    )
