import os
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Shard, Replicate, DTensor

def test_identity():
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29512"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    torch.distributed.init_process_group(backend="gloo")
    mesh = init_device_mesh("cpu", (1,))

    x = torch.randn(2, 4, requires_grad=True)
    dx = distribute_tensor(x, mesh, [Shard(0)])
    
    dy = dx * 1.0
    dy.sum().backward()
    
    print(f"dx.grad: {dx.grad}")
    print(f"x.grad: {x.grad}")

    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    test_identity()
