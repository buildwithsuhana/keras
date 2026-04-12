import os
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, distribute_tensor

# Mocking a distributed environment
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29509"
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"

try:
    torch.distributed.init_process_group(backend="gloo", rank=0, world_size=1)
    mesh = init_device_mesh("cpu", (1,))
    
    t = torch.randn(8, 4)
    dt = distribute_tensor(t, mesh, [Replicate()])
    
    print(f"Is DTensor: {isinstance(dt, DTensor)}")
    
    p = torch.nn.Parameter(dt)
    
    print("Testing all_reduce on p.data.to_local()")
    try:
        torch.distributed.all_reduce(p.data.to_local(), op=torch.distributed.ReduceOp.SUM)
        print("all_reduce on p.data.to_local() succeeded!")
    except Exception as e:
        print(f"all_reduce on p.data.to_local() failed with error: {e}")

finally:
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
