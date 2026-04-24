import torch
from torch.distributed.tensor import DTensor, DeviceMesh, Shard, Replicate
try:
    from torch.distributed.tensor.experimental import register_sharding
except ImportError:
    print("register_sharding NOT available")
    exit(0)

print("register_sharding is available")

# Setup a dummy mesh
mesh = DeviceMesh("cpu", torch.arange(1))

# A dummy tensor
t = torch.randn(4, 2)
dt = DTensor.from_local(t, mesh, [Replicate()])

# Register unbind for DTensor using register_sharding
@register_sharding(torch.ops.aten.unbind.int)
def unbind_sharding(tensor_spec, dim=0):
    # This is a bit complex to implement correctly, but for testing we can just
    # say it always replicates on output if we want.
    # The actual signature for unbind sharding is:
    # returns a sequence of 2-tuples (output_placements, input_placements)
    # But wait, unbind returns a SEQUENCE of tensors.
    # How does register_sharding handle multiple outputs?
    
    # Actually, unbind.int is:
    # aten.unbind.int(Tensor self, int dim=0) -> Tensor[]
    
    # For testing, let's just see if it works with one possibility
    # (Output placements, Input placements)
    # Output is a list of tensors.
    return [([Replicate()] * tensor_spec.shape[dim], [Replicate()])]

# Now try to unbind
try:
    results = torch.unbind(dt, 0)
    print("Unbind worked!")
    print("Result types:", [type(r) for r in results])
    print("Result placements:", [r.placements for r in results if isinstance(r, DTensor)])
except Exception as e:
    print(f"Unbind failed: {e}")

# Now try iteration
try:
    for x in dt:
        print("Iteration worked!")
        print("Item type:", type(x))
        break
except Exception as e:
    print(f"Iteration failed: {e}")
