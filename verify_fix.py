import os
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Shard, Replicate, DTensor
from keras.src.backend.torch.nn import dot_product_attention

def verify():
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29509"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="gloo")

    mesh = init_device_mesh("cpu", (1,))

    q = torch.randn(2, 32, 12, 64)
    k = torch.randn(2, 32, 12, 64)
    v = torch.randn(2, 32, 12, 64)
    
    dq = distribute_tensor(q, mesh, [Shard(2)])
    dk = distribute_tensor(k, mesh, [Shard(2)])
    dv = distribute_tensor(v, mesh, [Shard(2)])
    
    print("Calling dot_product_attention (should now work internally)...")
    from torch.nn.attention import sdpa_kernel, SDPBackend
    with sdpa_kernel(SDPBackend.MATH):
        try:
            out = dot_product_attention(dq, dk, dv)
            print("Success!")
            print(f"Output shape: {out.shape}")
            print(f"Output placements: {out.placements}")
        except Exception as e:
            print(f"Failed: {e}")
            import traceback
            traceback.print_exc()

    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    verify()
