# Fix Plan: DTensor/Tensor Mixing Error

## Problem Analysis

The error `RuntimeError: aten.sub.Tensor: got mixed torch.Tensor and DTensor, need to convert all torch.Tensor to DTensor before calling distributed operators!` occurs during training when operations mix regular `torch.Tensor` and `DTensor` objects.

### Root Cause
1. Weights are properly distributed as DTensors
2. But during forward/backward passes, intermediate tensors (activations, gradients) might be regular torch.Tensors
3. When DTensor weights interact with regular torch.Tensor inputs, PyTorch throws this error

### Solution
Add tensor conversion utilities in `distribution_lib.py` that:
1. Convert regular tensors to replicated DTensors when needed
2. Use `torch.distributed.tensor.parallel.parallelize_module` for automatic DTensor handling
3. Provide helper functions for ensuring all tensors in operations are properly typed

## Implementation Plan

### Step 1: Add `ensure_dtensor` helper function
```python
def ensure_dtensor(tensor, device_mesh=None, placements=None):
    """Ensure tensor is a DTensor. Convert if not."""
```

### Step 2: Add `parallelize_module` integration
Use `torch.distributed.tensor.parallel.parallelize_module` for automatic DTensor handling in layers

### Step 3: Modify `distribute_variable` to handle the conversion
Ensure that all variable operations use proper tensor types

### Step 4: Add tensor conversion in core.py
Update Variable operations to handle DTensor conversion

## Files to Modify

1. `keras/src/backend/torch/distribution_lib.py`:
   - Add `ensure_dtensor()` function
   - Add `ensure_dtensor_input()` function  
   - Add `parallelize_torch_module()` integration
   - Update `distribute_variable()` to handle mixed tensor scenarios

2. `keras/src/backend/torch/core.py`:
   - Update Variable class to handle DTensor conversions
   - Add `_ensure_dtensor()` method to Variable

## Implementation Details

### ensure_dtensor function
```python
def ensure_dtensor(tensor, device_mesh=None, placements=None):
    """Convert a tensor to DTensor if it isn't already.
    
    Args:
        tensor: torch.Tensor or DTensor
        device_mesh: DeviceMesh to use for conversion
        placements: Placements for the DTensor (default: Replicate)
    
    Returns:
        DTensor (or original if already a DTensor)
    """
```

### ensure_dtensor_input function  
Used for layer inputs to ensure they're DTensors when the layer has DTensor weights:

```python
def ensure_dtensor_input(input_tensor, device_mesh):
    """Ensure input tensor is a DTensor for DTensor operations.
    
    When a layer has DTensor weights but receives regular tensor inputs,
    PyTorch DTensor operations will fail. This helper converts regular
    tensors to replicated DTensors.
    """
```

### parallelize_module Integration

Use PyTorch's `parallelize_module` to automatically handle:
- Converting inputs to DTensors
- Sharding weights according to layout
- Converting outputs back to local tensors

This is the most robust solution as it lets PyTorch handle all the tensor conversions.

## Testing

After implementation, test with:
```bash
torchrun --nproc_per_node=2 kaggle_distributed_test.py
```

Expected behavior:
- All operations should complete without "mixed torch.Tensor and DTensor" error
- Weights should still be properly sharded
- Training should complete successfully

