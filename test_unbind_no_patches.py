import torch
from torch.distributed.tensor import DTensor, DeviceMesh, Shard, Replicate

# Initialize process group
torch.distributed.init_process_group(backend='gloo', rank=0, world_size=1, init_method='tcp://127.0.0.1:23457')

# Setup a dummy mesh
mesh = DeviceMesh("cpu", torch.arange(1))

# A dummy tensor
t = torch.randn(4, 2)
dt = DTensor.from_local(t, mesh, [Replicate()])

print("Attempting unbind WITHOUT any patches...")
try:
    results = torch.unbind(dt, 0)
    print("Unbind worked natively!")
    print("Result types:", [type(r) for r in results])
except Exception as e:
    print(f"Unbind failed natively: {e}")

print("Attempting iteration WITHOUT any patches...")
try:
    for x in dt:
        print("Iteration worked natively!")
        break
except Exception as e:
    print(f"Iteration failed natively: {e}")
