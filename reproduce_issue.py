import os
import torch
from keras.src.backend.torch import distribution_lib

def test_to_backend_device():
    # Mocking cuda availability to test both cases
    original_cuda_is_available = torch.cuda.is_available
    
    print("Testing with CUDA available mocked as True")
    torch.cuda.is_available = lambda: True
    os.environ["LOCAL_RANK"] = "0"
    device = distribution_lib._to_backend_device("cpu")
    print(f"Requested 'cpu', got: {device}")
    
    os.environ["LOCAL_RANK"] = "1"
    device = distribution_lib._to_backend_device(None)
    print(f"Requested None (LOCAL_RANK=1), got: {device}")

    print("\nTesting with CUDA available mocked as False")
    torch.cuda.is_available = lambda: False
    device = distribution_lib._to_backend_device("gpu:0")
    print(f"Requested 'gpu:0', got: {device}")
    
    # Restore
    torch.cuda.is_available = original_cuda_is_available

if __name__ == "__main__":
    test_to_backend_device()
