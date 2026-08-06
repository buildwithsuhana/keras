import os

import torch
from torch.distributed.tensor import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor import Replicate
from torch.overrides import TorchFunctionMode

# Initialize distributed
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"
if not torch.distributed.is_initialized():
    torch.distributed.init_process_group(backend="gloo", rank=0, world_size=1)


class MyFunctionMode(TorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        print(f"Intercepted function: {func.__name__}")
        if kwargs is None:
            kwargs = {}
        return func(*args, **kwargs)


device_mesh = DeviceMesh("cpu", torch.arange(1))
dtensor = DTensor.from_local(torch.randn(2, 2), device_mesh, [Replicate()])

print("--- Starting op: + ---")
with MyFunctionMode():
    dtensor + 1
