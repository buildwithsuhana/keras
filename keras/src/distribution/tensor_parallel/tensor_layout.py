import collections
import numpy as np
from keras.src import ops

def split_tensor_for_parallelism(tensor, index, device_count, dim):
    """Calculates a slice of a tensor along a specified dimension for a
    given index.

    This utility is used in tensor parallelism API to distribute a
    tensor across multiple devices.

    Args:
        tensor: The full tensor to be sharded.
        index: The index of the device/shard to return (e.g., 0, 1, 2...).
        device_count: The total number of parallel devices or splits.
        dim: The dimension along which to split the tensor. If -1, the
            last dimension is used.

    Returns:
        A tensor slice corresponding to the given `index`.
    """
    
    if hasattr(tensor, 'numpy'):
        tensor_cpu = tensor.numpy()
    else:
        tensor_cpu = np.array(tensor)

    if dim == -1:
        split_dim = tensor_cpu.ndim - 1
    else:
        split_dim = dim

    splits = np.array_split(
        tensor_cpu, indices_or_sections=device_count, axis=split_dim
    )
    return splits[index]


LayoutMap = collections.namedtuple("LayoutMap", ["state_rules", "output_rules"])