import torch
from torch.distributed.tensor import DTensor, Replicate
from torch.distributed.device_mesh import init_device_mesh

if not torch.distributed.is_initialized():
    torch.distributed.init_process_group(
        "gloo", rank=0, world_size=1, init_method="tcp://127.0.0.1:0"
    )

mesh = init_device_mesh("cpu", (1,))
v = DTensor.from_local(torch.ones(2, 2), device_mesh=mesh, placements=(Replicate(),))

print(f"isinstance(v, torch.Tensor): {isinstance(v, torch.Tensor)}")

print("Testing if v:")
try:
    if v:
        print("v is True")
    else:
        print("v is False")
except Exception as e:
    print(f"if v failed: {e}")

print("\nTesting if v is not None:")
try:
    if v is not None:
        print("v is not None")
except Exception as e:
    print(f"if v is not None failed: {e}")

print("\nTesting bool(v):")
try:
    print(bool(v))
except Exception as e:
    print(f"bool(v) failed: {e}")
