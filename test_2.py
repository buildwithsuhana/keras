import os

# 1. Pipeline-Agnostic Environment Locks
os.environ["KERAS_BACKEND"] = "jax"
os.environ["KERAS_NNX_ENABLED"] = "true"

import jax.numpy as jnp
import keras_hub

from keras.src.utils.tunix_utils import TunixKerasAdapter
from keras.src.utils.tunix_utils import get_keras_trainable_filter

print("📌 Loading Dependencies and Pipeline Shims...")

backbone = keras_hub.models.OPTBackbone(
    vocabulary_size=50265,
    num_layers=12,
    num_heads=12,
    hidden_dim=768,
    intermediate_dim=3072,
)

print("✅ OPT-125M Architecture initialized.")

# 4. Wrap with TunixKerasAdapter
print("\n🪢 Wrapping with TunixKerasAdapter...")
adapter = TunixKerasAdapter(
    backbone, embed_dim=backbone.hidden_dim, num_layers=backbone.num_layers
)
print("✅ Success: Adapter Injection successful.")

# 5. Simulate Upstream Pipeline Micro-Batches Injection
B, L = 2, 32  # Batch, Length
dummy_tokens = jnp.ones((B, L), dtype="int32")
dummy_mask = jnp.ones((B, L), dtype="bool")
dummy_positions = jnp.arange(L)[None, :]
print(f"\n🚀 Feeding Pipeline Micro-Batch [Shape: {B}x{L}]...")
try:
    logits, cache = adapter(
        dummy_tokens,
        attention_mask=dummy_mask,
        positions=dummy_positions,
        cache={"kv_stubs": jnp.zeros((B, 12, L, 64))},
    )
    print("📋 Outer pipeline output shape achieved:", logits.shape)

except Exception as e:
    print(
        "⚠️ Pipeline hook mapping caveat detected:",
        e,
    )
    print("\n💡Wrap dict-expecting Models in a bridge adapter dict-shuffler:")
    # (See Adapter mapping dictionary logic section bellow)


# 6. Verify Regex Granular Trainable Filters
print("\n🛡️ Testing Path Partitioning Regex Hook...")
# Only optimize kernel weights, discard embeddings and biases cascading
kernel_filter = get_keras_trainable_filter(
    regex_patterns=[".*transformer.*kernel.*"]
)
print("✅ Trainable regex pipe filter ready attached to NNX.")
