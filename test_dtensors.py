import os
os.environ["KERAS_BACKEND"] = "torch"

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor

def test_mix_dtensors():
    if not dist.is_initialized():
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "12357"
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        dist.init_process_group(backend="gloo")

    mesh = init_device_mesh("cpu", (1,))
    
    t1 = torch.ones((2, 2))
    dt_sharded = distribute_tensor(t1, mesh, [Shard(0)])
    
    t2 = torch.ones((2, 2))
    dt_replicated = distribute_tensor(t2, mesh, [Replicate()])
    
    print(f"Testing minimum(Sharded DTensor, Replicated DTensor):")
    try:
        res = torch.minimum(dt_sharded, dt_replicated)
        print("  Success")
        print(f"  Result type: {type(res)}")
        print(f"  Result placements: {res.placements}")
    except Exception as e:
        print(f"  Failed: {e}")

    dist.destroy_process_group()

if __name__ == "__main__":
    test_mix_dtensors()
