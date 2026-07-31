import torch

from keras.src.backend.torch import core

print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"torch.backends.mps.is_available(): {torch.backends.mps.is_available()}")
print(f"core.get_device(): {core.get_device()}")
