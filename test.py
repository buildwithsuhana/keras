import os
from typing import Any
from typing import Optional
from typing import Tuple

# Set environment variables before importing Keras/JAX
os.environ["KERAS_BACKEND"] = "jax"
os.environ["KERAS_NNX_ENABLED"] = "true"

import jax.numpy as jnp
import optax
from absl import app
from flax import nnx

import keras


class TunixKerasAdapter(nnx.Module):
    """Bridges Keras models to Tunix Trainer expectations."""

    def __init__(self, keras_model: keras.Model, pad_id: Optional[int] = None):
        super().__init__()
        self.base = keras_model
        self._pad_id = pad_id

    def __call__(
        self,
        input_tokens: jnp.ndarray,
        positions: jnp.ndarray,
        cache: Optional[Any] = None,
        attention_mask: Optional[Any] = None,
        decoder_segment_ids: Optional[Any] = None,
        output_hidden_states: bool = False,
    ) -> Tuple[jnp.ndarray, Optional[Any]]:
        """Matches Tunix model call signature."""

        # Synthesize segment IDs if missing and pad_id is provided
        if decoder_segment_ids is None and self._pad_id is not None:
            decoder_segment_ids = (input_tokens != self._pad_id).astype(
                jnp.int32
            )

        # Forward call to Keras model
        logits = self.base(input_tokens)

        # Tunix expects (logits, cache) tuple
        return logits, None


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    print("Initializing Keras model in NNX mode...")

    # Create a tiny Keras model
    inputs = keras.Input(shape=(10,), dtype="int32")
    embedding = keras.layers.Embedding(input_dim=100, output_dim=16)(inputs)
    outputs = keras.layers.Dense(100)(embedding)  # Output logits over vocab
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Wrap it in the Adapter
    print("Wrapping with TunixKerasAdapter...")
    adapter = TunixKerasAdapter(model, pad_id=0)

    # Generate dummy data
    batch_size = 2
    seq_len = 10
    input_tokens = (
        jnp.arange(batch_size * seq_len).reshape((batch_size, seq_len)) % 100
    )
    positions = jnp.broadcast_to(jnp.arange(seq_len), (batch_size, seq_len))

    # Dummy targets (logits)
    targets = jnp.ones((batch_size, seq_len, 100)) * 0.1

    # Setup Optimizer
    # We optimize all variables in this simple demo
    optimizer = nnx.Optimizer(adapter, optax.adam(1e-3), wrt=nnx.Variable)

    def loss_fn(model, x, pos, y):
        logits, _ = model(x, pos)
        # Simple Mean Squared Error for dummy training
        loss = jnp.mean((logits - y) ** 2)
        return loss

    @nnx.jit
    def train_step(model, optimizer, x, pos, y):
        # We must explicitly specify nnx.Variable because Keras variables
        # in NNX mode inherit from nnx.Variable, not nnx.Param (the default).
        grad_fn = nnx.value_and_grad(
            loss_fn, argnums=nnx.DiffState(0, nnx.Variable)
        )
        loss, grads = grad_fn(model, x, pos, y)
        optimizer.update(model, grads)
        return loss

    print("\n--- Starting Training Loop ---")
    for step in range(10):
        loss = train_step(adapter, optimizer, input_tokens, positions, targets)
        print(f"Step {step}: Loss = {loss.item():.6f}")

    print("\n--- Success! ---")


if __name__ == "__main__":
    app.run(main)
