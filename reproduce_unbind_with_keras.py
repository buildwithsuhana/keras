import os
os.environ["KERAS_BACKEND"] = "torch"

import torch
from torch.distributed.tensor import DTensor, DeviceMesh, Shard, Replicate

# Mock distributed environment
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"
if not torch.distributed.is_initialized():
    torch.distributed.init_process_group(backend="gloo", rank=0, world_size=1)

print(f"Before Keras import, DTensor.unbind: {DTensor.unbind}")

import keras
from keras.src.backend.torch import distribution_lib

print(f"After Keras import, DTensor.unbind: {DTensor.unbind}")
print(f"distribution_lib._dtensor_unbind_patched: {distribution_lib._dtensor_unbind_patched}")

if DTensor.unbind == distribution_lib._dtensor_unbind_patched:
    print("SUCCESS: DTensor.unbind IS PATCHED")
else:
    print("FAILURE: DTensor.unbind IS NOT PATCHED")
