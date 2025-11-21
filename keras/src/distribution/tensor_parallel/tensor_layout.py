import collections

from keras.src import ops


# tensor_layout.py

import numpy as np
from keras.src import ops

def split_tensor_for_parallelism(tensor, index, device_count, dim):
    """
    Splits tensor using Host Memory (NumPy) to avoid OOM on GPU.
    """
    # PILLAR 1: Ensure we are working with Host Memory (NumPy)
    # This prevents allocating the full tensor on the GPU.
    if hasattr(tensor, "numpy"):
        tensor_data = tensor.numpy()
    elif hasattr(tensor, "value"): 
        tensor_data = ops.convert_to_numpy(tensor.value)
    else:
        tensor_data = ops.convert_to_numpy(tensor)

    if dim == -1:
        split_dim = tensor_data.ndim - 1
    else:
        split_dim = dim

    # Perform the split on CPU RAM
    splits = np.array_split(
        tensor_data, indices_or_sections=device_count, axis=split_dim
    )
    
    # Return the specific slice (still as a NumPy array)
    return splits[index]


LayoutMap = collections.namedtuple("LayoutMap", ["state_rules", "output_rules"])