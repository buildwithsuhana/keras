import os

os.environ["KERAS_BACKEND"] = "jax"
os.environ["KERAS_NNX_ENABLED"] = "true"

import jax.numpy as jnp

import keras
from keras.src.utils.tunix_utils import TunixKerasAdapter

# 1. Build a functional model taking a LIST of inputs
tokens_in = keras.Input(shape=(None,), dtype="int32", name="input_tokens")
mask_in = keras.Input(shape=(None,), dtype="bool", name="input_mask")

x = keras.layers.Embedding(input_dim=1000, output_dim=128)(tokens_in)
x = keras.layers.Dense(64)(x)

model_list_inputs = keras.Model(inputs=[tokens_in, mask_in], outputs=x)

# 2. Wrap with Adapter (Routes by positional index in list)
adapter = TunixKerasAdapter(model_list_inputs)

# 3. Simulate Pipeline Call
B, L = 2, 32
tokens = jnp.ones((B, L), dtype="int32")
mask = jnp.ones((B, L), dtype="bool")

# 🚀 Adapter wraps tokens and mask into `[tokens, mask]` automatically.
logits, cache = adapter(tokens, attention_mask=mask)
print("List-Input Model Logits Shape:", logits.shape)
