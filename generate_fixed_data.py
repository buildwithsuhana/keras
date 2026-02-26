import os
import numpy as np

# Set seed for numpy
np.random.seed(42)

# Generate fixed inputs
batch_size = 2
seq_len = 32
vocab_size = 50272

token_ids = np.random.randint(0, vocab_size, (batch_size, seq_len)).astype("int32")
padding_mask = np.ones((batch_size, seq_len), dtype="int32")
# Use int32 for categorical targets, but ensure model weights are float64
targets = np.random.randint(0, vocab_size, (batch_size, seq_len)).astype("int32")

np.save("token_ids.npy", token_ids)
np.save("padding_mask.npy", padding_mask)
np.save("targets.npy", targets)

print("Fixed inputs saved to .npy files.")
