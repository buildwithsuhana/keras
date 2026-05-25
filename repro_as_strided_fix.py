import os
os.environ["KERAS_BACKEND"] = "torch"

import torch
import keras
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Shard

def test_as_strided_fix():
    os.environ.update({
        "MASTER_ADDR": "localhost",
        "MASTER_PORT": "29564",
        "RANK": "0",
        "WORLD_SIZE": "1",
    })
    
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="gloo")

    # Initialize Keras distribution which should trigger _register_distributed_strategies
    mesh = keras.distribution.DeviceMesh((1,), ("model",), ["cpu:0"])
    layout_map = keras.distribution.LayoutMap(mesh)
    dist = keras.distribution.ModelParallel(layout_map=layout_map, batch_dim_name="model")
    keras.distribution.set_distribution(dist)

    # Now test if aten.as_strided works via torch.compile
    # We use a real DeviceMesh for DTensor
    torch_mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("model",))
    x = torch.randn(4, 4)
    dt = distribute_tensor(x, torch_mesh, [Shard(0)])
    
    # Slicing often uses as_strided in compiled graphs
    @torch.compile
    def func(t):
        return t[1:3, 1:3]
    
    try:
        out = func(dt)
        print("Successfully ran compiled function with as_strided!")
        print(f"Output type: {type(out)}")
        print(f"Output: {out}")
    except Exception as e:
        print(f"Failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_as_strided_fix()
