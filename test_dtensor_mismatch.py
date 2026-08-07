import os

import torch
from torch.distributed.tensor import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor import Replicate

# Mock distributed environment
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29515"
if not torch.distributed.is_initialized():
    torch.distributed.init_process_group(backend="gloo", rank=0, world_size=1)

# We can't easily create a CUDA mesh without CUDA.
# But we can check what happens when mesh and tensor device mismatch.

mesh = DeviceMesh("cpu", torch.arange(1))
# Create a tensor on another "device" - well, we only have CPU.
# But what if we use a mesh of type "cuda" (if we could)?

print("Testing mismatch between tensor device and mesh device type")
# Since I can't easily test CUDA, let's just see if from_local moves it.

local_tensor = torch.randn(2, 2)
dtensor = DTensor.from_local(local_tensor, mesh, [Replicate()])
print(f"Mesh device type: {mesh.device_type}")
print(f"Tensor device: {local_tensor.device}")
print(f"DTensor device: {dtensor.device}")
