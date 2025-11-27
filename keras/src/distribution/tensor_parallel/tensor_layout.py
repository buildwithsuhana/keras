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
    if dim == -1:
        split_dim = len(tensor.shape) - 1
    else:
        split_dim = dim

    # [CRITICAL FIX] Force conversion to standard NumPy array (Host Memory).
    # This is essential for large models (like Gemma 9B).
    # Without this, JAX attempts to load the FULL tensor (e.g. 1.71GB embedding)
    # onto the default TPU device (TPU 0) just to slice it. Since TPU 0 is 
    # already holding shards, it runs out of memory (RESOURCE_EXHAUSTED).
    # Converting to 'np.array' forces the slice to execute on the CPU.
    tensor_numpy = np.array(tensor)

    shape = tensor_numpy.shape
    total_length = shape[split_dim]
    shard_size = total_length // device_count
    
    start_idx = index * shard_size
    
    # Handle remainder for the last shard
    if index == device_count - 1:
        end_idx = total_length
    else:
        end_idx = (index + 1) * shard_size
        
    # Construct the slice tuple: (:, :, start:end, :)
    slice_indices = [slice(None)] * len(shape)
    slice_indices[split_dim] = slice(start_idx, end_idx)
    
    # Perform the slice on Host RAM and return
    # This returns a small chunk (e.g. ~200MB) that fits easily on the TPU.
    return tensor_numpy[tuple(slice_indices)]


LayoutMap = collections.namedtuple("LayoutMap", ["state_rules", "output_rules"])