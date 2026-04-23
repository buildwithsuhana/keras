import os
os.environ["KERAS_BACKEND"] = "torch"
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Shard, Replicate, DTensor
import keras

def test_mixed_dtensor():
    if not torch.distributed.is_initialized():
        # Fake initialization for single process test if possible, 
        # but DTensor usually needs real distributed env.
        # Let's try to mock what's needed or just use the existing mp.py failure.
        pass

    # This is hard to test without a real distributed setup.
    # But I can see the code in keras/src/backend/torch/numpy.py
    
    x1 = torch.ones((2, 2))
    mesh = init_device_mesh("cpu", (1,))
    dt1 = distribute_tensor(x1, mesh, [Replicate()])
    
    x2 = torch.ones((2, 2))
    
    try:
        res = torch.minimum(dt1, x2)
        print("Success")
    except Exception as e:
        print(f"Caught expected error: {e}")

if __name__ == "__main__":
    test_mixed_dtensor()
