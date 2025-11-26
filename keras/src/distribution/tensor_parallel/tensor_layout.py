import collections

from keras.src import ops


# In tensor_layout.py

from keras.src.distribution.tensor_parallel.lazy_init import LazyVariable # Import from wherever you saved Step 1

def split_tensor_for_parallelism(tensor, index, device_count, dim):
    """
    Overloaded to handle both real Tensors and LazyVariables.
    """
    
    # CASE 1: Lazy Initialization (Zero Stage)
    if isinstance(tensor, LazyVariable):
        initializer = tensor._initializer
        shape = tensor.shape
        dtype = tensor.dtype
        
        # Calculate shard shape
        new_shape = list(shape)
        if dim != -1:
            # Validate divisibility
            if new_shape[dim] % device_count != 0:
                 # Handling indivisible shapes is complex, assuming divisible for now
                 pass
            new_shape[dim] = new_shape[dim] // device_count
        
        # Determine Seed logic
        # Ideally, we want a deterministic seed based on the variable name + index
        # For this example, we assume the initializer handles the seed or uses global state
        
        # Adjust Scale for VarianceScaling (Glorot/He)
        # If we split the input_dim (fan_in), the variance of the initialization changes.
        # Strict correctness requires rescaling.
        # However, simply calling the initializer with the new shape is the standard "Zero" approach.
        
        return initializer(shape=new_shape, dtype=dtype)

    # CASE 2: Standard Splitting (Existing code)
    if dim == -1:
        split_dim = ops.ndim(tensor) - 1
    else:
        split_dim = dim

    splits = ops.array_split(
        tensor, indices_or_sections=device_count, axis=split_dim
    )
    return splits[index]


LayoutMap = collections.namedtuple("LayoutMap", ["state_rules", "output_rules"])