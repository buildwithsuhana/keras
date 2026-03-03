import os
import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor, DeviceMesh, Replicate, distribute_tensor

# Initialize process group for single process
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12355"
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
dist.init_process_group(backend="gloo")

# Setup a dummy mesh (single process)
mesh = DeviceMesh("cpu", torch.arange(1))

# Create a DTensor
t1 = torch.ones((2, 2))
dt1 = distribute_tensor(t1, mesh, [Replicate()])

# Create a vanilla Tensor
t2 = torch.ones((2, 2))

print(f"dt1 type: {type(dt1)}")
print(f"t2 type: {type(t2)}")

try:
    res = dt1 + t2
    print("dt1 + t2 worked!")
    print(f"Result type: {type(res)}")
except Exception as e:
    print(f"dt1 + t2 failed: {e}")

try:
    res = torch.add(dt1, t2)
    print("torch.add(dt1, t2) worked!")
    print(f"Result type: {type(res)}")
except Exception as e:
    print(f"torch.add(dt1, t2) failed: {e}")

try:
    res = torch.zeros_like(dt1)
    print("torch.zeros_like(dt1) worked!")
    print(f"Result type: {type(res)}")
except Exception as e:
    print(f"torch.zeros_like(dt1) failed: {e}")

try:
    res = dt1 * 2.0
    print("dt1 * 2.0 worked!")
    print(f"Result type: {type(res)}")
except Exception as e:
    print(f"dt1 * 2.0 failed: {e}")
