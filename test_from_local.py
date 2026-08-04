import os

import torch
import torch.distributed as dist
from torch.distributed.tensor import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor import Replicate


def test_from_local_cpu_to_cuda():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"

    dist.init_process_group("nccl")
    torch.cuda.set_device(0)

    mesh = DeviceMesh("cuda", [0])

    # Create CPU tensor
    local_tensor = torch.ones((2, 2), device="cpu")
    print(f"Local tensor device: {local_tensor.device}")

    try:
        # Try from_local with CPU tensor on CUDA mesh
        dt = DTensor.from_local(local_tensor, mesh, [Replicate()])
        print(f"DTensor created! Device: {dt.device}")
    except Exception as e:
        print(f"DTensor.from_local failed with CPU tensor: {e}")

    dist.destroy_process_group()


if __name__ == "__main__":
    test_from_local_cpu_to_cuda()
