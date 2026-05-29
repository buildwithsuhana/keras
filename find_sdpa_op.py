import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

def find_sdpa_op():
    q = torch.randn(1, 1, 1, 1)
    k = torch.randn(1, 1, 1, 1)
    v = torch.randn(1, 1, 1, 1)
    
    class Tracker(torch.Tensor):
        @classmethod
        def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
            print(f"OP: {func}")
            return func(*args, **kwargs)
    
    qt = q.as_subclass(Tracker)
    kt = k.as_subclass(Tracker)
    vt = v.as_subclass(Tracker)
    
    with sdpa_kernel(SDPBackend.MATH):
        try:
            F.scaled_dot_product_attention(qt, kt, vt)
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    find_sdpa_op()
