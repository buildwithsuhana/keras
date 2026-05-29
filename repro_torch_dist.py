import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def run(rank, size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group('gloo', rank=rank, world_size=size)
    print(f"Rank {rank} initialized")
    dist.destroy_process_group()

if __name__ == "__main__":
    size = 2
    mp.spawn(run, args=(size,), nprocs=size, join=True)
