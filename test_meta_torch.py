import os
os.environ["KERAS_BACKEND"] = "torch"
import torch
import keras
from torch.distributed.tensor import DeviceMesh, DTensor, Replicate, Shard, distribute_tensor

# Fake a distributed environment if not already initialized
if not torch.distributed.is_initialized():
    torch.distributed.init_process_group(backend="gloo", rank=0, world_size=1, init_method="tcp://127.0.0.1:23456")

mesh = DeviceMesh("cpu", [0])

# Create a meta tensor
meta_tensor = torch.ones((10, 10), device="meta")

# Distribute it
# PyTorch 2.x distribute_tensor supports meta tensors
dt = distribute_tensor(meta_tensor, mesh, [Shard(0)])
print(f"DTensor on meta: {dt.device}")
print(f"Is meta: {dt.is_meta}")

# Move to CPU
dt_cpu = dt.to("cpu")
print(f"DTensor on CPU: {dt_cpu.device}")
print(f"Is meta: {dt_cpu.is_meta}")
print(f"Local shard shape: {dt_cpu.to_local().shape}")
print(f"Local shard values: {dt_cpu.to_local()}") # Should be uninitialized or something?
