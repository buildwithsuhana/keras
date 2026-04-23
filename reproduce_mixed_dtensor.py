import os
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor

def reproduce():
    # Initialize process group for a single process (mocking distributed)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29505"
    dist.init_process_group("gloo", rank=0, world_size=1)
    
    mesh = init_device_mesh("cpu", (1,))
    
    # Create a DTensor
    t1 = torch.ones((2, 32), device="cpu")
    dt1 = distribute_tensor(t1, mesh, [Replicate()])
    
    # Create a plain tensor
    t2 = torch.ones((1, 32), device="cpu")
    
    print("Testing torch.minimum(DTensor, aligned torch.Tensor)")
    try:
        if isinstance(dt1, DTensor) and not isinstance(t2, DTensor):
            t2_dt = DTensor.from_local(t2, dt1.device_mesh, [Replicate()] * dt1.device_mesh.ndim)
        res = torch.minimum(dt1, t2_dt)
        print("Success!")
        print(f"Result type: {type(res)}")
    except Exception as e:
        print(f"Failed with error: {e}")

    dist.destroy_process_group()

if __name__ == "__main__":
    reproduce()
