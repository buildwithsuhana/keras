import os
os.environ["KERAS_BACKEND"] = "torch"

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, distribute_tensor

def test_mix():
    if not dist.is_initialized():
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "12356"
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        dist.init_process_group(backend="gloo")

    mesh = init_device_mesh("cpu", (1,))
    
    t1 = torch.ones((2, 2))
    dt1 = distribute_tensor(t1, mesh, [Replicate()])
    
    t2 = torch.ones((2, 2))
    
    print(f"Testing add(DTensor, Tensor):")
    try:
        res = torch.add(dt1, t2)
        print("  Success")
    except Exception as e:
        print(f"  Failed: {e}")

    print(f"Testing minimum(DTensor, Tensor):")
    try:
        res = torch.minimum(dt1, t2)
        print("  Success")
    except Exception as e:
        print(f"  Failed: {e}")

    dist.destroy_process_group()

if __name__ == "__main__":
    test_mix()
