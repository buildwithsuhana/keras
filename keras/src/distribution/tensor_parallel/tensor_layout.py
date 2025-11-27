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
    # Resolve negative dimension index
    if dim == -1:
        split_dim = len(tensor.shape) - 1
    else:
        split_dim = dim

    # [OOM FIX] Use direct slicing instead of ops.array_split.
    # ops.array_split creates ALL splits at once (e.g., 8 copies for 8 GPUs),
    # which causes a massive memory spike.
    # Direct slicing only materializes the specific shard we need.
    
    shape = tensor.shape
    total_length = shape[split_dim]
    shard_size = total_length // device_count
    
    start_idx = index * shard_size
    
    # Handle remainder for the last shard (mimics numpy.array_split behavior)
    if index == device_count - 1:
        end_idx = total_length
    else:
        end_idx = (index + 1) * shard_size
        
    # Construct the slice tuple: (:, :, start:end, :)
    slice_indices = [slice(None)] * len(shape)
    slice_indices[split_dim] = slice(start_idx, end_idx)
    
    # Perform the slice. 
    # If 'tensor' is on CPU (recommended), this returns a CPU tensor/view.
    return tensor[tuple(slice_indices)]


LayoutMap = collections.namedtuple("LayoutMap", ["state_rules", "output_rules"])