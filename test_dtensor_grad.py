import os
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Shard, Replicate, DTensor

class SDPA_DTensor_Full(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value):
        # inputs are DTensors
        ctx.save_for_backward(query, key, value)
        ql = query._local_tensor
        out_l = ql * 1.0
        return DTensor.from_local(out_l, query.device_mesh, query.placements)

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output is a DTensor
        query, key, value = ctx.saved_tensors
        ql = grad_output._local_tensor
        return DTensor.from_local(ql, grad_output.device_mesh, grad_output.placements), None, None

def test_full_custom_grad():
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29514"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="gloo")
    mesh = init_device_mesh("cpu", (1,))

    x = torch.randn(2, 4, requires_grad=True)
    dx = distribute_tensor(x, mesh, [Shard(0)])
    
    dy = SDPA_DTensor_Full.apply(dx, None, None)
    
    dy.sum().backward()
    print(f"dx.grad type: {type(dx.grad)}")
    print(f"dx.grad: {dx.grad}")
    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    test_full_custom_grad()
