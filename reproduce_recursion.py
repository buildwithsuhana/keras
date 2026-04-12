import os
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

# Mocking TorchLayer and TorchModuleWrapper behavior
class MockTorchLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.variables = []

    def train(self, mode=True):
        print(f"Calling train on {self.__class__.__name__}")
        return super().train(mode)

class KerasModuleWrapper(nn.Module):
    def __init__(self, keras_model):
        super().__init__()
        self.__dict__["_keras_model"] = keras_model
        # In real Keras, it registers parameters here

    def forward(self, *args, **kwargs):
        return self._keras_model(*args, **kwargs)

class TorchModuleWrapper(MockTorchLayer):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def train(self, mode=True):
        print(f"Calling train on TorchModuleWrapper")
        self.module.train(mode)
        return super().train(mode)

def reproduce():
    model = MockTorchLayer()
    
    # Simulating what happens in TorchTrainer._setup_ddp
    wrapper = KerasModuleWrapper(model)
    
    # In a real DDP setup, we need process group initialized
    # But we can just mock DDP or use a simple wrapper that calls train()
    
    # self._ddp_model = DDP(wrapper, ...)
    # Here we just use a simple wrapper to simulate DDP's behavior of calling train on its module
    class MockDDP(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def train(self, mode=True):
            print("Calling train on MockDDP")
            self.module.train(mode)
            return super().train(mode)

    ddp_model = MockDDP(wrapper)
    
    # This is what TorchLayer._setattr_hook does:
    # it wraps the ddp_model in a TorchModuleWrapper because it's a Module but not a Layer
    model._ddp_model = TorchModuleWrapper(ddp_model)
    
    print("Starting train()...")
    try:
        model.train()
    except RecursionError:
        print("Caught expected RecursionError!")

if __name__ == "__main__":
    reproduce()
