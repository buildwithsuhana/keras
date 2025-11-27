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
    # 1. Resolve negative dimensions (e.g. -1)
    if dim == -1:
        split_dim = len(tensor.shape) - 1
    else:
        split_dim = dim

    # [CRITICAL FIX] Force conversion to standard NumPy array.
    # This is the MAGIC LINE that fixes the OOM.
    # It pulls the data to Host RAM (CPU) so we don't use TPU memory
    # as a scratchpad for the full 1.7GB tensor.
    tensor_numpy = np.array(tensor)

    # 2. Calculate slice indices on CPU
    shape = tensor_numpy.shape
    total_length = shape[split_dim]
    shard_size = total_length // device_count
    
    start_idx = index * shard_size
    
    # Handle remainder for the last shard (if any)
    if index == device_count - 1:
        end_idx = total_length
    else:
        end_idx = (index + 1) * shard_size
        
    # 3. Create the slice object
    slice_indices = [slice(None)] * len(shape)
    slice_indices[split_dim] = slice(start_idx, end_idx)
    
    # 4. Perform slicing on CPU memory
    # This returns a small ~215MB chunk (1.7GB / 8) which fits easily on TPU.
    return tensor_numpy[tuple(slice_indices)]


LayoutMap = collections.namedtuple("LayoutMap", ["state_rules", "output_rules"])