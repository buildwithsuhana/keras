import torch
from torch.distributed.tensor import DTensor, DeviceMesh, Shard, Replicate

# Initialize process group
if not torch.distributed.is_initialized():
    torch.distributed.init_process_group(backend='gloo', rank=0, world_size=1, init_method='tcp://127.0.0.1:23459')

# Setup a dummy mesh
mesh = DeviceMesh("cpu", torch.arange(1))

# A dummy tensor sharded on dim 0
t = torch.randn(4, 2)
dt = DTensor.from_local(t, mesh, [Shard(0)])

# ONLY patch unbind
DTensor.unbind = lambda self, dim=0: self.to_local().unbind(dim)

print("Attempting iteration with ONLY unbind patched...")
try:
    for x in dt:
        print("Iteration worked!")
        break
except Exception as e:
    print(f"Iteration failed: {e}")
