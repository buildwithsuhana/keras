import torch
from torch.distributed.tensor import DTensor, DeviceMesh, Shard, Replicate

# Try to patch torch.Tensor.view
original_view = torch.Tensor.view

def patched_view(self, *args, **kwargs):
    print(f"Patched view called on {type(self)}")
    return original_view(self, *args, **kwargs)

try:
    torch.Tensor.view = patched_view
    x = torch.ones(2, 2)
    x.view(4)
    print("Successfully patched torch.Tensor.view")
except Exception as e:
    print(f"Failed to patch torch.Tensor.view: {e}")

# Try to patch torch.reshape
original_reshape = torch.reshape

def patched_reshape(input, shape):
    print(f"Patched reshape called on {type(input)}")
    return original_reshape(input, shape)

try:
    torch.reshape = patched_reshape
    x = torch.ones(2, 2)
    torch.reshape(x, (4,))
    print("Successfully patched torch.reshape")
except Exception as e:
    print(f"Failed to patch torch.reshape: {e}")
