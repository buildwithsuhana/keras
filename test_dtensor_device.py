import os

import torch
from torch.distributed.tensor import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor import Replicate

# Mock distributed environment
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29505"
if not torch.distributed.is_initialized():
    torch.distributed.init_process_group(backend="gloo", rank=0, world_size=1)

device_type = "cpu"
if torch.cuda.is_available():
    device_type = "cuda"
    torch.cuda.set_device(0)

print(f"Using device type: {device_type}")

mesh = DeviceMesh(device_type, torch.arange(1))
local_tensor = torch.randn(2, 2).cpu()  # Force CPU

try:
    dtensor = DTensor.from_local(local_tensor, mesh, [Replicate()])
    print("Success: DTensor.from_local(cpu_tensor, cuda_mesh) worked")
    print(f"DTensor device: {dtensor.device}")
except Exception as e:
    print(f"Failed: DTensor.from_local(cpu_tensor, cuda_mesh) raised: {e}")

local_tensor_on_device = local_tensor.to(device_type)
try:
    dtensor = DTensor.from_local(local_tensor_on_device, mesh, [Replicate()])
    print("Success: DTensor.from_local(device_tensor, device_mesh) worked")
    print(f"DTensor device: {dtensor.device}")
except Exception as e:
    print(f"Failed: DTensor.from_local(device_tensor, device_mesh) raised: {e}")
