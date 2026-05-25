import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Shard, Replicate
import os

def test_as_strided():
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29562"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="gloo")
        
    mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("model",))
    
    x = torch.randn(4, 4)
    sharding = [Shard(0)]
    dt = distribute_tensor(x, mesh, sharding)
    
    print(f"Original DTensor: {dt}")
    
    # This might trigger aten.as_strided in some versions or via torch.compile
    def func(t):
        # Slicing often uses as_strided under the hood in compiled graphs
        return t[1:3, 1:3]
    
    compiled_func = torch.compile(func)
    try:
        out = compiled_func(dt)
        print(f"Output: {out}")
    except Exception as e:
        print(f"Caught expected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_as_strided()
