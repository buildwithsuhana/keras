import collections
from keras.src import ops

import numpy as np
from keras.src import ops

def split_tensor_for_parallelism(tensor, index, device_count, dim):
    """
    Splits a tensor. Handles both Real tensors and Ghost (Lazy) tensors.
    """
    # 1. Check if it is a Ghost Variable
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
    
    # 2. GHOST PATH: Allocate NEW memory on the specific device
    if is_ghost:
        new_shape = list(tensor_shape)
        new_shape[split_dim] = shard_size
        new_shape_tuple = tuple(new_shape)
        
        if hasattr(tensor, "initializer") and tensor.initializer is not None:
            try:
                return tensor.initializer(shape=new_shape_tuple, dtype=tensor.dtype)
            except Exception:
                return ops.zeros(new_shape_tuple, dtype=tensor.dtype)
        else:
            return ops.zeros(new_shape_tuple, dtype=tensor.dtype)

    # 3. REAL TENSOR PATH (The Fix)
    # -------------------------------------------------------------------------
    # CRITICAL CHANGE: Move data to CPU (numpy) before splitting.
    # This prevents doubling VRAM usage on the GPU during the split.
    # -------------------------------------------------------------------------
    
    # Force conversion to standard Numpy (CPU RAM)
    # ops.convert_to_numpy is robust across JAX/Torch/TF
    cpu_tensor = ops.convert_to_numpy(tensor)
    
    # Perform the split on CPU
    # usage: numpy.array_split (NOT ops.array_split) to ensure CPU processing
    splits = np.array_split(
        cpu_tensor, indices_or_sections=device_count, axis=split_dim
    )
    
    # The result is a numpy array. When returned, Keras will automatically 
    # cast it back to the target GPU device during assignment.
    return splits[index]

LayoutMap = collections.namedtuple("LayoutMap", ["state_rules", "output_rules"])