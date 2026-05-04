import os
import torch
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")
t = torch.tensor([1.0])
print(f"Tensor device: {t.device}")

# If we try to force CPU
torch.set_default_device('cpu')
t2 = torch.tensor([1.0])
print(f"Tensor 2 device: {t2.device}")
