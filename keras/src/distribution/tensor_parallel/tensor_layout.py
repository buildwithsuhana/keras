import collections
import numpy as np  # <--- ADD THIS
from keras.src import ops

def split_tensor_for_parallelism(tensor, index, device_count, dim):
    """
    Splits a tensor. Handles both Real tensors and Ghost (Lazy) tensors.
    """
    # 1. Check if it is a Ghost Variable (Duck typing to avoid circular import)
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
    
    # 2. If it is a Ghost, we allocate NEW memory on the specific device
    if is_ghost:
        new_shape = list(tensor_shape)
        new_shape[split_dim] = shard_size
        new_shape_tuple = tuple(new_shape)
        
        # Use the original initializer if available
        if hasattr(tensor, "initializer") and tensor.initializer is not None:
            try:
                # Call the original initializer with the new SHARDED shape
                return tensor.initializer(shape=new_shape_tuple, dtype=tensor.dtype)
            except Exception:
                # Fallback if initializer fails (e.g. complex state)
                return ops.zeros(new_shape_tuple, dtype=tensor.dtype)
        else:
            # Default fallback
            return ops.zeros(new_shape_tuple, dtype=tensor.dtype)

    # 3. Standard logic for Real tensors
    # --- CRITICAL FIX START ---
    # We use NumPy (CPU) for splitting to prevent OOM.
    # ops.array_split() would force the full tensor onto the TPU, causing the crash.
    
    if hasattr(tensor, "numpy"):
        # If it's a Keras/TF/JAX tensor, pull it to CPU RAM
        tensor_cpu = tensor.numpy()
    else:
        # If it's a list or other object, wrap it
        tensor_cpu = np.array(tensor)

    splits = np.array_split(
        tensor_cpu, indices_or_sections=device_count, axis=split_dim
    )
    # --- CRITICAL FIX END ---
    
    return splits[index]

LayoutMap = collections.namedtuple("LayoutMap", ["state_rules", "output_rules"])