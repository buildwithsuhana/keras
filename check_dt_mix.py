import os

import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate
from torch.distributed.tensor import distribute_tensor

os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29505"

torch.distributed.init_process_group("gloo")
mesh = init_device_mesh("cpu", (1,))

t = torch.ones((4, 4))
dt = distribute_tensor(t, mesh, [Replicate()])

try:
    res = dt + t
    print(f"DTensor + Tensor success: {type(res)}")
except Exception as e:
    print(f"DTensor + Tensor failed: {e}")

try:
    res = torch.cat([dt, t])
    print(f"torch.cat([DTensor, Tensor]) success: {type(res)}")
except Exception as e:
    print(f"torch.cat([DTensor, Tensor]) failed: {e}")

torch.distributed.destroy_process_group()
