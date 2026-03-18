import numpy as np
import os

# Generate fixed float64 data for high-precision comparison
np.random.seed(42)
batch_size = 4
seq_len = 32
x_token_ids = np.random.randint(0, 50272, size=(batch_size, seq_len)).astype("int32")
x_padding_mask = np.ones((batch_size, seq_len), dtype="int32")
# Generate double precision target
y = np.random.randn(batch_size, seq_len, 768).astype("float64")

os.makedirs("data_cmp", exist_ok=True)
np.save("data_cmp/x_token_ids.npy", x_token_ids)
np.save("data_cmp/x_padding_mask.npy", x_padding_mask)
np.save("data_cmp/y.npy", y)
print("Comparison data saved to data_cmp/")
