import numpy as np

class LayoutMap(dict):
    """
    A flexible dictionary-based LayoutMap for manual Tensor Parallelism.
    Inherits from dict to allow arbitrary attribute storage (output_rules).
    """
    def __init__(self, device_mesh=None):
        super().__init__()
        self.device_mesh = device_mesh
        # Stores communication rules for layers (e.g. 'allreduce sum')
        self.output_rules = {}

    @property
    def state_rules(self):
        """Alias self to match Keras distribution API expectation."""
        return self

def split_tensor_for_parallelism(tensor, index, device_count, dim):
    """
    Slices a tensor for Tensor Parallelism on the CPU.
    """
    # 1. Resolve negative dimensions
    ndim = len(tensor.shape)
    if dim < 0:
        split_dim = ndim + dim
    else:
        split_dim = dim

    # [OOM FIX] Force conversion to Host RAM (NumPy).
    # We use np.array explicitly. 
    # NOTE: If input is bfloat16, this might cast to float32 in standard NumPy.
    # We accept this temporarily because we will immediately push it to GPU 
    # and free the CPU memory in parameter_sharding.py.
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