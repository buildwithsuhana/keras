import torch
import os

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    x = torch.ones((10,), device="cuda")
    try:
        y = torch.prod(x)
        print(f"Result: {y}")
    except Exception as e:
        print(f"Failed with error: {e}")
else:
    print("CUDA not available, cannot test.")
