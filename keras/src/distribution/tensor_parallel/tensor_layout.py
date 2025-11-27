import numpy as np

try:
    import ml_dtypes
    HAS_ML_DTYPES = True
except ImportError:
    HAS_ML_DTYPES = False

_ML_DTYPES_LOGGED = False

class LayoutMap(dict):
    def __init__(self, device_mesh=None):
        super().__init__()
        self.device_mesh = device_mesh
        self.output_rules = {}

    @property
    def state_rules(self):
        return self

def split_tensor_for_parallelism(tensor, index, device_count, dim):
    global _ML_DTYPES_LOGGED
    if not _ML_DTYPES_LOGGED:
        print(f"   ℹ️ [Layout] ML_DTYPES available: {HAS_ML_DTYPES}")
        _ML_DTYPES_LOGGED = True

    ndim = len(tensor.shape)
    if dim < 0: split_dim = ndim + dim
    else: split_dim = dim

    if hasattr(tensor, "numpy"):
        tensor_numpy = tensor.numpy()
    else:
        tensor_numpy = np.array(tensor)

    total_length = tensor_numpy.shape[split_dim]
    shard_size = total_length // device_count
    
    start_idx = index * shard_size
    if index == device_count - 1: end_idx = total_length
    else: end_idx = (index + 1) * shard_size
        
    slices = [slice(None)] * ndim
    slices[split_dim] = slice(start_idx, end_idx)
    
    shard = tensor_numpy[tuple(slices)]

    if HAS_ML_DTYPES:
        return shard.astype(ml_dtypes.bfloat16)
    else:
        return shard.astype(np.float16)