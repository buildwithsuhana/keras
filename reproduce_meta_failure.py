import torch

from keras.src.backend.torch.core import Variable
from keras.src.backend.torch.core import device_scope
from keras.src.backend.torch.core import get_device
from keras.src.distribution.distribution_lib import (
    DeviceMesh as KerasDeviceMesh,
)
from keras.src.distribution.distribution_lib import LayoutMap
from keras.src.distribution.distribution_lib import ModelParallel

# Setup torch distributed
if not torch.distributed.is_initialized():
    torch.distributed.init_process_group(
        backend="gloo",
        rank=0,
        world_size=1,
        init_method="tcp://127.0.0.1:23456",
    )

mesh = KerasDeviceMesh((1,), ["data"], ["cpu"])
layout_map = LayoutMap(mesh)
dist = ModelParallel(layout_map=layout_map, batch_dim_name="data")

with dist.scope():
    with device_scope("meta"):
        print(f"Current device according to get_device(): {get_device()}")

        # Variable might be initialized on cpu if not careful
        v = Variable(torch.ones((2, 3)))
        print(f"v device: {v.value.device}")

        x = torch.ones((2, 3), device=get_device())
        print(f"x device: {x.device}")

        try:
            z = x * v.value
            print(f"z device: {z.device}")
        except Exception as e:
            print(f"Operation failed: {e}")
