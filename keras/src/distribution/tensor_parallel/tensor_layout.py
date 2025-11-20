import collections

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
    if dim == -1:
        split_dim = ops.ndim(tensor) - 1
    else:
        split_dim = dim

    # --- MINIMAL OOM FIX START ---
    # Replaced ops.array_split with direct slicing to avoid materializing
    # unneeded shards for other devices in memory.
    
    shape = tensor.shape
    total_length = shape[split_dim]
    shard_size = total_length // device_count
    
    start = index * shard_size
    
    # Handle potential remainders on the last device
    if index == device_count - 1:
        end = total_length
    else:
        end = start + shard_size

    # Create a slice object: [:, :, start:end, :]
    slices = [slice(None)] * len(shape)
    slices[split_dim] = slice(start, end)
    
    return tensor[tuple(slices)]
    # --- MINIMAL OOM FIX END ---


LayoutMap = collections.namedtuple("LayoutMap", ["state_rules", "output_rules"])