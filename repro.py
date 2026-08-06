import os

import torch
import torch.distributed
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Replicate
from torch.distributed.tensor import distribute_tensor


def test():
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29507"
    torch.distributed.init_process_group(backend="gloo", rank=0, world_size=1)

    mesh = DeviceMesh("cpu", torch.arange(1))
    print(f"Mesh created: {mesh}")

    tensor = torch.ones((2, 2))
    dtensor = distribute_tensor(tensor, mesh, [Replicate()])
    print(f"DTensor created: {dtensor}")

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    test()
