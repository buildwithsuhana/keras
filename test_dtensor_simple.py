import os
import torch
import numpy as np
import subprocess
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Shard, Replicate

def run(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["GLOO_SOCKET_IFNAME"] = "lo0"
    os.environ["TP_SOCKET_IFNAME"] = "lo0"
    
    torch.distributed.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    print(f"[Rank {rank}] Group initialized")
    
    mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("batch", "model"))
    print(f"[Rank {rank}] Mesh initialized: {mesh}")
    
    # Test sharding
    big_tensor = torch.randn(8, 8)
    sharded_tensor = distribute_tensor(big_tensor, mesh, [Replicate(), Shard(1)])
    print(f"[Rank {rank}] Sharded tensor: {sharded_tensor}")
    
    torch.distributed.destroy_process_group()
    print(f"[Rank {rank}] Done")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        run(int(sys.argv[1]), int(sys.argv[2]))
    else:
        world_size = 4
        processes = []
        for rank in range(world_size):
            p = subprocess.Popen([sys.executable, __file__, str(rank), str(world_size)])
            processes.append(p)
        for p in processes:
            p.wait()
