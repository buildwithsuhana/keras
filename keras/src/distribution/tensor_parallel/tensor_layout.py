import logging
import numpy as np

# Try to import ml_dtypes for bfloat16 support on CPU
try:
    import ml_dtypes
    HAS_ML_DTYPES = True
except ImportError:
    HAS_ML_DTYPES = False

class LayoutMap(dict):
    """
    A flexible dictionary-based LayoutMap for manual Tensor Parallelism.
    Inherits from dict to allow arbitrary attribute storage (output_rules).
    """
    def __init__(self, device_mesh=None):
        super().__init__()
        self.device_mesh = device_mesh
        self.output_rules = {}

    @property
    def state_rules(self):
        return self

def split_tensor_for_parallelism(tensor, index, device_count, dim):
    """
    Slices a tensor on CPU and downcasts to bfloat16 to save Device Memory.
    """
    # 1. Resolve negative dimensions
    ndim = len(tensor.shape)
    if dim < 0:
        split_dim = ndim + dim
    else:
        split_dim = dim

    # 2. Move to Host RAM (NumPy)
    # NOTE: This usually promotes bfloat16 -> float32
    if hasattr(tensor, "numpy"):
        tensor_numpy = tensor.numpy()
    else:
        tensor_numpy = np.array(tensor)

    # 3. Calculate slice indices
    total_length = tensor_numpy.shape[split_dim]
    shard_size = total_length // device_count
    
    start_idx = index * shard_size
    if index == device_count - 1:
        end_idx = total_length
    else:
        end_idx = (index + 1) * shard_size
        
    slices = [slice(None)] * ndim
    slices[split_dim] = slice(start_idx, end_idx)
    
    # 4. Perform Slice
    shard = tensor_numpy[tuple(slices)]

    # 5. [CRITICAL OOM FIX] Downcast on CPU before sending to GPU
    # This prevents the backend from allocating a massive float32 buffer on VRAM.
    if HAS_ML_DTYPES:
        # Convert float32 -> bfloat16 (numpy compatible)
        return shard.astype(ml_dtypes.bfloat16)
    else:
        # Fallback to float16 if ml_dtypes is missing (saves memory, slight precision risk)
        return shard.astype(np.float16)