
import os
os.environ["KERAS_BACKEND"] = "torch"
os.environ["KERAS_TORCH_DEVICE"] = "cpu"
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12357"

import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate
import keras
from keras.src.backend.torch.numpy import argmax as knp_argmax
import numpy as np

def test_dtensor_argmax():
    # Initialize a single-device mesh on CPU
    mesh = init_device_mesh("cpu", (1,))
    
    # Create a local tensor
    local_tensor = torch.tensor([[1, 5, 2], [4, 3, 6]]).float()
    
    # Create a DTensor by replicating the local tensor
    placements = [Replicate()]
    dt_x = DTensor.from_local(local_tensor, mesh, placements)
    
    print(f"Input type: {type(dt_x)}")
    
    # Enable torch sharding scope to trigger _maybe_promote_to_dtensor in cast/convert_to_tensor
    from keras.src.backend.torch.distribution_lib import sharding_scope
    
    with sharding_scope():
        # Call knp_argmax
        res = knp_argmax(dt_x, axis=1)
    
    print(f"Result: {res}")
    print(f"Result type: {type(res)}")
    
    expected = torch.tensor([1, 2])
    print(f"Expected: {expected}")
    
    # If it's a DTensor, convert to local for comparison
    if isinstance(res, DTensor):
        res_local = res.to_local()
    else:
        res_local = res
        
    assert torch.all(res_local == expected)
    print("Test passed!")

if __name__ == "__main__":
    try:
        test_dtensor_argmax()
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
