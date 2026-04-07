
import torch
from torch.distributed.tensor import DTensor, DeviceMesh, Shard, Replicate
import os

# Mock distributed environment if not already set
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"
if not torch.distributed.is_initialized():
    torch.distributed.init_process_group(backend="gloo", rank=0, world_size=1)

device_mesh = DeviceMesh("cpu", [1])
tensor = torch.arange(4).reshape(2, 2)
dtensor = DTensor.from_local(tensor, device_mesh, [Replicate()])

print(f"DTensor type: {type(dtensor)}")

# Check if unbind exists and if it's patched
print(f"DTensor.unbind: {DTensor.unbind}")

try:
    print("Testing iteration...")
    for i, t in enumerate(dtensor):
        print(f"Iteration {i}, type: {type(t)}")
except Exception as e:
    print(f"Iteration failed: {e}")

try:
    print("Testing unbind directly...")
    unbounded = dtensor.unbind(0)
    print(f"Unbind success, type of first element: {type(unbounded[0])}")
except Exception as e:
    print(f"Unbind failed: {e}")
