import os
os.environ["KERAS_BACKEND"] = "torch"
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12360"

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, distribute_tensor

def test_mixed_dtensor_tensor():
    dist.init_process_group(backend="gloo")
    
    device = "cpu"
    mesh = init_device_mesh(device, (1,))
    
    local_tensor = torch.tensor([1.0, 2.0])
    # distribute_tensor might need to be called on all ranks
    dtensor = distribute_tensor(local_tensor, mesh, [Replicate()])
    
    regular_tensor = torch.tensor([10.0, 20.0])
    
    print(f"DTensor type: {type(dtensor)}")
    print(f"Regular Tensor type: {type(regular_tensor)}")
    
    try:
        # Test addition
        result = dtensor + regular_tensor
        print(f"Result type (DTensor + Tensor): {type(result)}")
        print(f"Result: {result}")
        
        # Test reversed addition
        result_rev = regular_tensor + dtensor
        print(f"Result type (Tensor + DTensor): {type(result_rev)}")
        print(f"Result: {result_rev}")
        
    except Exception as e:
        print(f"Mixed operation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_mixed_dtensor_tensor()
