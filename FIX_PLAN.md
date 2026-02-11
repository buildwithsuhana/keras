# Fix Plan: Device Mismatch Error in DTensor Model Parallelism

## Root Cause Analysis

The error occurs in `TokenAndPositionEmbedding.call()` when adding:
```python
outputs = embedded_tokens + embedded_positions
```

Error: `Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cuda:1!`

### Why it happens:

1. **Sharded Embeddings**: When model parallelism is enabled with `axes=('model', None)` for embeddings:
   - Token embedding weights are sharded across the 'model' axis (devices)
   - Position embedding weights are also sharded

2. **Device Placement Issues**:
   - `embedded_tokens` = `token_embedding(inputs)` → tensor on the embedding lookup device
   - `embedded_positions` = `position_embedding(embedded_tokens, ...)` → involves:
     - `ops.convert_to_tensor(self.position_embeddings)` - DTensor → local tensor
     - `ops.slice` - might create tensor on different device  
     - `ops.broadcast_to` - broadcasts but device handling is inconsistent
   - Result: `embedded_tokens` and `embedded_positions` end up on different devices

3. **Addition Operation Fails**: PyTorch's `+` operator requires both tensors on the same device

## Files to Modify

### 1. `keras/src/backend/torch/numpy.py`
**Issue**: `broadcast_to` doesn't handle DTensor properly

**Fix**: Modify `broadcast_to` to ensure consistent device placement
```python
def broadcast_to(x, shape):
    x = convert_to_tensor(x)
    # For DTensor, ensure broadcast maintains proper device placement
    from torch.distributed.tensor import DTensor
    if isinstance(x, DTensor):
        # Ensure x is on the correct device before broadcasting
        x_local = x._local_tensor
        if x_local.device != x.device():
            x_local = x_local.to(x.device())
        result = torch.broadcast_to(x_local, shape)
        return DTensor.from_local(result, x._spec, reshape=False)
    return torch.broadcast_to(x, shape)
```

### 2. `keras/src/backend/torch/numpy.py`
**Issue**: `slice` might create tensors on wrong device for DTensor

**Fix**: Ensure slice maintains device placement for DTensor
```python
def slice(inputs, start_indices, shape):
    # Existing code...
    inputs = convert_to_tensor(inputs)
    
    from torch.distributed.tensor import DTensor
    if isinstance(inputs, DTensor):
        # Get local tensor and slice it
        local_tensor = inputs._local_tensor
        slices = [
            python_slice(start_index, start_index + length)
            for start_index, length in zip(start_indices, shape)
        ]
        sliced_local = local_tensor[tuple(slices)]
        # Return DTensor with same spec
        return DTensor.from_local(sliced_local, inputs._spec, reshape=False)
    
    # Original code for non-DTensor...
```

### 3. `keras/src/backend/torch/core.py`
**Issue**: `convert_to_tensor` might not preserve DTensor properties

**Fix**: Add explicit handling for DTensor in `convert_to_tensor`
```python
def convert_to_tensor(x, dtype=None, sparse=None, ragged=None):
    # Early return for DTensor - already a proper tensor
    from torch.distributed.tensor import DTensor
    if isinstance(x, DTensor):
        if dtype is not None:
            local_dtype = to_torch_dtype(dtype)
            if x._local_tensor.dtype != local_dtype:
                new_local = x._local_tensor.to(dtype=local_dtype)
                return DTensor.from_local(new_local, x._spec)
        return x
    
    # ... rest of existing code
```

### 4. Alternative Fix in keras-hub (if needed)
**File**: `keras-hub/keras_hub/src/layers/modeling/token_and_position_embedding.py`

**Fix**: Add explicit device sync before addition
```python
def call(self, inputs, start_index=0, positions=None):
    embedded_tokens = self.token_embedding(inputs)
    embedded_positions = self.position_embedding(
        embedded_tokens,
        start_index=start_index,
        positions=positions,
    )
    
    # FIX: Ensure both tensors are on the same device
    if hasattr(embedded_tokens, 'device') and hasattr(embedded_positions, 'device'):
        if embedded_tokens.device != embedded_positions.device:
            embedded_positions = embedded_positions.to(embedded_tokens.device)
    
    outputs = embedded_tokens + embedded_positions
    # ... rest of method
```

## Implementation Order

1. **Priority 1**: Fix `broadcast_to` in `numpy.py` - This is the most likely culprit
2. **Priority 2**: Fix `slice` in `numpy.py` - Used by position embedding slicing
3. **Priority 3**: Fix `convert_to_tensor` in `core.py` - Ensure DTensor preservation
4. **Priority 4**: Fix `token_and_position_embedding.py` - Device sync as fallback

## Testing

After implementing fixes, test with:
```python
CUDA_VISIBLE_DEVICES=0,1 python test_torch_model_parallel_2gpu.py
```

Expected: No device mismatch error, training completes successfully.

## Notes

- The DTensor handling should be conditional on `torch.distributed.tensor` being available
- Use `_local_tensor` and `_spec` attributes to properly handle DTensor
- The fixes should be backward compatible with non-DTensor tensors

