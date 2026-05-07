import os
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Shard, Replicate, DTensor

# Mock environment for DTensor
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29506"
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
torch.distributed.init_process_group(backend="gloo")

mesh = init_device_mesh("cpu", (1,))

# (batch, seq, heads, head_dim)
t = torch.randn(2, 32, 12, 64)
# shard on heads (dim 2)
dt = distribute_tensor(t, mesh, [Shard(2)])
print(f"Original placements: {dt.placements}")

dt_t = torch.transpose(dt, 1, 2)
print(f"Transposed shape: {dt_t.shape}")
print(f"Transposed placements: {dt_t.placements}")

torch.distributed.destroy_process_group()
