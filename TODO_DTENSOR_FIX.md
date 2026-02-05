# TODO List for DTensor Mixed Tensor Fix

## Task: Fix "got mixed torch.Tensor and DTensor" error in distributed training

### Files Modified:
1. `keras/src/backend/torch/core.py`
   - ✅ Fixed indexing warning: `x[seq]` → `x[tuple(seq)]` at line 705-713

2. `keras/src/backend/torch/numpy.py`
   - ✅ Added DTensor handling to `matmul()` - handles mixed DTensor/torch.Tensor operands
   - ✅ Added DTensor handling to `multiply()` - handles mixed DTensor/torch.Tensor operands  
   - ✅ Added DTensor handling to `dot()` - handles mixed DTensor/torch.Tensor operands
   - ✅ Added DTensor handling to `einsum()` - handles mixed DTensor/torch.Tensor operands
   - ✅ Added DTensor handling to `tensordot()` - handles mixed DTensor/torch.Tensor operands

### Implementation Summary:

The fix follows the pattern already established in the `add()` function:
1. Check if either operand is a DTensor
2. If one is DTensor and the other is regular tensor, convert the regular tensor to DTensor
3. If device_mesh is None (symbolic build), fall back to extracting local tensors

This ensures that PyTorch distributed operations always receive either:
- Both operands as DTensors (normal case for distributed training)
- Both operands as regular tensors (fallback for symbolic build)

### Key Functions Fixed:
- `matmul()` - Used by Dense layers for matrix multiplication
- `multiply()` - Element-wise multiplication
- `dot()` - Dot product
- `einsum()` - Einstein summation (used in attention mechanisms)
- `tensordot()` - Tensor dot product (used in attention mechanisms)

### Testing:
Run the hybrid DP+MP test to verify the fix:
```bash
python kaggle_hybrid_dp_mp_input_fix.py
```

