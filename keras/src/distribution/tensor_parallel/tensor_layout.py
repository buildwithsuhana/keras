import numpy as np

class LayoutMap(dict):
    """
    A dictionary-based LayoutMap that stores sharding rules.
    Replaces the namedtuple to allow mutable item assignment.
    """
    def __init__(self, device_mesh=None):
        super().__init__()
        self.device_mesh = device_mesh
        self.output_rules = {}

    @property
    def state_rules(self):
        """Alias for compatibility with sharding strategies."""
        return self

def split_tensor_for_parallelism(tensor, index, device_count, dim):
    """
    Slices a tensor for Tensor Parallelism.
    CRITICAL: Performs slicing on CPU (NumPy) to avoid Accelerator OOM.
    """
    # 1. Resolve negative dimensions
    ndim = len(tensor.shape)
    if dim < 0:
        split_dim = ndim + dim
    else:
        split_dim = dim

    # [OOM FIX] Force conversion to Host RAM (NumPy) immediately.
    # If we don't do this, the backend might try to load the full tensor 
    # onto the TPU to perform the slice op, causing instant OOM.
    # We detach, move to CPU, and cast to numpy.
    if hasattr(tensor, "numpy"):
        tensor_numpy = tensor.numpy()
    else:
        tensor_numpy = np.array(tensor)

    # 2. Calculate slice indices
    total_length = tensor_numpy.shape[split_dim]
    shard_size = total_length // device_count
    
    start_idx = index * shard_size
    if index == device_count - 1:
        end_idx = total_length
    else:
        end_idx = (index + 1) * shard_size
        
    # 3. Create slicing tuple
    slices = [slice(None)] * ndim
    slices[split_dim] = slice(start_idx, end_idx)
    
    # 4. Return the specific shard (small chunk)
    return tensor_numpy[tuple(slices)]