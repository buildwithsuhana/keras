import os
import torch
import torch.distributed as dist
from torch.distributed._tensor import DeviceMesh, DTensor, Replicate, Shard, Partial, distribute_tensor

def setup_dist():
    if not dist.is_initialized():
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "12356"
        os.environ["RANK"] = os.environ.get("RANK", "0")
        os.environ["WORLD_SIZE"] = os.environ.get("WORLD_SIZE", "1")
        dist.init_process_group(backend="gloo")

setup_dist()
rank = dist.get_rank()
world_size = dist.get_world_size()

mesh = DeviceMesh("cpu", list(range(world_size)))

# Create a tensor and shard it such that it becomes Partial after some op
# Or just manually create a Partial tensor for testing
local_tensor = torch.ones(4, 4) * (rank + 1)
# Each rank has a different part of the sum. 
# Global sum would be sum(1..world_size) * ones(4,4)
partial_tensor = DTensor.from_local(local_tensor, mesh, [Partial()])

# Apply dropout with a fixed seed
torch.manual_seed(42)
try:
    # Test if native dropout handles Partial
    dropped = torch.nn.functional.dropout(partial_tensor, p=0.5, training=True)
    print(f"Rank {rank}: Native dropout success.")
    
    # Check if sum(dropped) matches dropout(sum)
    replicated = dropped.redistribute(mesh, [Replicate()])
    if rank == 0:
        print(f"Rank 0: Result after all_reduce (replicated):\n{replicated.to_local()}")
except Exception as e:
    print(f"Rank {rank}: Native dropout failed: {e}")

dist.destroy_process_group()
