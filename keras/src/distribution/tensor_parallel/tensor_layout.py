import collections
import numpy as np  # <--- 1. ADD THIS IMPORT

from keras.src import ops

def split_tensor_for_parallelism(tensor, index, device_count, dim):
    """
    Splits a tensor. Handles both Real tensors and Ghost (Lazy) tensors.
    """
    # 1. Check if it is a Ghost Variable (Duck typing)
    is_ghost = hasattr(tensor, "shape") and "GhostVariable" in str(type(tensor))
    
    tensor_shape = tensor.shape
    
    if dim == -1:
        split_dim = len(tensor_shape) - 1
    else:
        split_dim = dim

    # Calculate the size of the shard
    full_dim_size = tensor_shape[split_dim]
    if full_dim_size % device_count != 0:
        raise ValueError(
            f"Dimension {split_dim} (size {full_dim_size}) is not divisible "
            f"by device_count {device_count}."
        )
    
    shard_size = full_dim_size // device_count
    
    # 2. If it is a Ghost, we allocate NEW UNINITIALIZED memory on the specific device
    if is_ghost:
        new_shape = list(tensor_shape)
        new_shape[split_dim] = shard_size
        
        # We return a standard Keras Tensor/Variable initialized with Zeros
        # This allocates memory on the CURRENT device (set by the loop in tensor_layout)
        return ops.zeros(tuple(new_shape), dtype=tensor.dtype)

    # 3. Standard logic for Real tensors
    # 🔴 OLD CODE (Causes OOM because ops.array_split uses the TPU):
    # splits = ops.array_split(
    #     tensor, indices_or_sections=device_count, axis=split_dim
    # )
    # return splits[index]

    # 🟢 NEW CODE (Fixes OOM by keeping the master weights on CPU):
    
    # Convert to standard NumPy array. This forces the data to live on the CPU RAM.
    # Even if 'tensor' was already on CPU, this ensures it doesn't accidentally 
    # move to TPU for the split operation.
    tensor_cpu = np.array(tensor) 
    
    splits = np.array_split(
        tensor_cpu, indices_or_sections=device_count, axis=split_dim
    )
    
    # We return a NumPy array shard. 
    # When this shard is assigned to a Variable later, JAX will automatically 
    # transfer JUST this small chunk to the specific TPU core.
    return splits[index]

LayoutMap = collections.namedtuple("LayoutMap", ["state_rules", "output_rules"])