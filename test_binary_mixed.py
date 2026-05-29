import os

os.environ["KERAS_BACKEND"] = "torch"
import torch
from torch.distributed.tensor import DeviceMesh
from torch.distributed.tensor import Replicate
from torch.distributed.tensor import distribute_tensor

import keras

if not torch.distributed.is_initialized():
    torch.distributed.init_process_group(
        "gloo", rank=0, world_size=1, init_method="tcp://127.0.0.1:0"
    )

mesh = DeviceMesh("cpu", [0])
t1 = torch.randn(4)
dt = distribute_tensor(t1, mesh, [Replicate()])

t2 = torch.randn(4)  # local tensor

print("Testing dt + t2 (local)...")
try:
    res = dt + t2
    print(f"Success! Result type: {type(res)}")
except Exception as e:
    print(f"Failed with: {e}")

print("Testing keras.ops.add(dt, t2)...")
try:
    res = keras.ops.add(dt, t2)
    print(f"Success! Result type: {type(res)}")
except Exception as e:
    print(f"Failed with: {e}")
