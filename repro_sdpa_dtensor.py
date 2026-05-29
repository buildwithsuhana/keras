import os

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29501"
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["KERAS_BACKEND"] = "torch"
os.environ["KERAS_TORCH_DEVICE"] = "cpu"

import torch
from torch.distributed._tensor import DeviceMesh, DTensor, distribute_tensor, Shard
import keras

def _scaled_dot_product_attention(
    query, key, value, mask, is_causal, scale, flash_attention
):
    if flash_attention:
        with torch.nn.attention.sdpa_kernel(
            backends=[torch.nn.attention.SDPBackend.FLASH_ATTENTION],
        ):
            return torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=mask,
                is_causal=is_causal,
                scale=scale,
            )
    if mask is not None:
        mask = mask.contiguous()
    return torch.nn.functional.scaled_dot_product_attention(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        attn_mask=mask,
        is_causal=is_causal,
        scale=scale,
    )

def simple_dot_product_attention(
    query, key, value, mask=None, is_causal=False, scale=None, flash_attention=False
):
    is_dtensor = isinstance(query, DTensor)
    if is_dtensor:
        q_l = query.to_local()
        k_l = key.to_local()
        v_l = value.to_local()
        m_l = mask.to_local() if mask is not None else None
        
        out_l = _scaled_dot_product_attention(q_l, k_l, v_l, m_l, is_causal, scale, flash_attention)
        return DTensor.from_local(out_l, query.device_mesh, query.placements)
    else:
        return _scaled_dot_product_attention(query, key, value, mask, is_causal, scale, flash_attention)

def test_dtensor_sdpa():
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="gloo")
    device_mesh = DeviceMesh("cpu", torch.arange(1))
    
    q = torch.randn(1, 2, 4, 8, requires_grad=True)
    k = torch.randn(1, 2, 4, 8, requires_grad=True)
    v = torch.randn(1, 2, 4, 8, requires_grad=True)
    
    layout = Shard(1)
    
    dq = distribute_tensor(q.detach(), device_mesh, [layout]).requires_grad_(True)
    dk = distribute_tensor(k.detach(), device_mesh, [layout]).requires_grad_(True)
    dv = distribute_tensor(v.detach(), device_mesh, [layout]).requires_grad_(True)
    
    print("Testing Simplified SDPA with DTensor...")
    try:
        out = simple_dot_product_attention(dq, dk, dv)
        print("Forward successful")
        loss = out.sum()
        loss.backward()
        print("Backward successful")
        
        if dq.grad is not None:
            print("Grad for DQ exists and has shape:", dq.grad.shape)
            print("Is DQ.grad a DTensor?", isinstance(dq.grad, DTensor))
    except Exception as e:
        print(f"Failed with: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dtensor_sdpa()
