import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import os

class Wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return self.model(x)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.ones(1))
    def forward(self, x):
        return x * self.param

def test():
    # Mock distributed environment
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.distributed.init_process_group("gloo", rank=0, world_size=1)

    model = MyModel()
    wrapper = Wrapper(model)
    ddp = DDP(wrapper)
    
    # This is what TorchTrainer does:
    object.__setattr__(model, "ddp", ddp)
    
    print("Calling model.train()...")
    try:
        model.train()
        print("Success!")
    except RecursionError:
        print("RecursionError encountered!")
    finally:
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    test()
