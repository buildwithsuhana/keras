import torch
from torch.overrides import TorchFunctionMode


class MyFunctionMode(TorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        print(f"Intercepted function: {func.__name__}")
        if kwargs is None:
            kwargs = {}
        return func(*args, **kwargs)


# Activate globally
with MyFunctionMode():
    torch.add(torch.tensor(1), torch.tensor(2))
