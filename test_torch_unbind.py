import torch
from torch.distributed.tensor import DTensor, DeviceMesh, Shard, Replicate

# Initialize process group
if not torch.distributed.is_initialized():
    torch.distributed.init_process_group(backend='gloo', rank=0, world_size=1, init_method='tcp://127.0.0.1:23460')

mesh = DeviceMesh("cpu", torch.arange(1))
t = torch.randn(4, 2)
dt = DTensor.from_local(t, mesh, [Shard(0)])

# ONLY patch DTensor.unbind
DTensor.unbind = lambda self, dim=0: self.to_local().unbind(dim)

print("Attempting torch.unbind with ONLY DTensor.unbind patched...")
try:
    torch.unbind(dt, 0)
    print("torch.unbind worked!")
except Exception as e:
    print(f"torch.unbind failed: {e}")
