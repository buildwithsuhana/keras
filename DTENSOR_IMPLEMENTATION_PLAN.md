# DTensor Implementation Plan for Keras Torch Backend

## Overview
Integrate PyTorch DTensor (`torch.distributed.tensor`) into the Keras Torch backend distribution system to provide automatic sharding semantics similar to JAX.

## Key Principles
1. **Shard variables** - both initially and every time assigning a new value
2. **Apply shardings to outputs** - for ModelParallel configuration
3. **Distribute tensors from datasets** - using DTensor's data distribution

With DTensor, operations on sharded tensors are automatically applied to all shards, and communication (like reductions) happens automatically.

## Implementation Steps

### Phase 1: Refactor `keras/src/backend/torch/distribution_lib.py`

#### 1.1 Import DTensor and create helper functions
```python
from torch.distributed.tensor import distribute_tensor, DTensor, DeviceMesh, Shard, Replicate
from torch.distributed import init_process_group
```

#### 1.2 Create converter functions
- `_to_dtensor_mesh(device_mesh)`: Convert Keras DeviceMesh to PyTorch DeviceMesh
- `_to_dtensor_layout(tensor_layout)`: Convert Keras TensorLayout to DTensor placements

#### 1.3 Update core functions
- `distribute_variable(value, layout)`: Return DTensor instead of manually sharded tensor
- `distribute_tensor(tensor, layout)`: Use DTensor for distribution
- `all_gather_variable(variable)`: Use DTensor's globalize() method
- `all_reduce(tensor, op, axis_name)`: Use DTensor's reduce() method
- `all_gather(tensor, axis, axis_name)`: Use DTensor's all_gather() method

### Phase 2: Update `keras/src/backend/torch/core.py`

#### 2.1 Modify Variable class
- Ensure Variable.value works with DTensor operations
- Handle DTensor attributes properly

### Phase 3: Handle Keras/PyTorch naming convention difference

Keras uses `/` separators (e.g., `dense/kernel`) while PyTorch uses `.` (e.g., `dense.weight`). 

#### 3.1 Create adapter in distribution_lib.py
```python
def _adapt_path(path):
    """Convert Keras path format to PyTorch format and vice versa.
    
    This adapter allows regex patterns to work regardless of the separator
    used in the variable path. Keras uses '/' (e.g., 'dense/kernel') while
    PyTorch uses '.' (e.g., 'dense.weight').
    
    The adapter checks both formats when matching layout_map keys to ensure
    compatibility with both Keras and PyTorch variable naming conventions.
    """
    return path

def _match_layout_map_key(variable_path, layout_map):
    """Match a variable path against layout map keys, handling both separators.
    
    Args:
        variable_path: The variable path (e.g., 'dense/kernel')
        layout_map: The LayoutMap with regex keys
        
    Returns:
        The matching layout or None
    """
    # Try direct match first
    if variable_path in layout_map:
        return layout_map[variable_path]
    
    # Try converting the path to PyTorch format and matching
    torch_path = variable_path.replace('/', '.')
    if torch_path in layout_map:
        return layout_map[torch_path]
    
    # Try original path with regex
    matching_keys = []
    for k in layout_map:
        if re.search(k, variable_path):
            matching_keys.append(k)
        # Also try PyTorch format for regex matching
        torch_k = k.replace('/', '.')
        if re.search(torch_k, torch_path):
            matching_keys.append(k)
    
    if len(matching_keys) > 1:
        raise ValueError(
            f"Path '{variable_path}' matches multiple layout "
            f"specification keys: {matching_keys}. Please make "
            "sure each tensor/variable path only matches at most "
            "one layout specification key in the LayoutMap."
        )
    elif len(matching_keys) == 1:
        return layout_map[matching_keys[0]]
    return None
```

### Phase 4: Update Layer distribution handling

In `keras/src/layers/layer.py`:
- Ensure `distribution_lib.distribute_tensor()` works with DTensor outputs

### Phase 5: Data distribution

In `keras/src/backend/torch/distribution_lib.py`:
- Update `distribute_data_input()` to use DTensor

### Phase 6: Testing

- Verify basic distribution works
- Test ModelParallel with DTensor
- Test DataParallel with DTensor
- Test all_gather_variable and all_reduce operations

## Files to Modify

1. `keras/src/backend/torch/distribution_lib.py` - Main changes
2. `keras/src/backend/torch/core.py` - Variable class adjustments
3. `keras/src/backend/torch/layer.py` - Possibly update if needed

## Compatibility Notes

- DTensor requires PyTorch 2.0+
- Need to check if DTensor is available before using it
- Graceful fallback if DTensor is not available

## Success Criteria

1. Variables are distributed using DTensor with correct sharding
2. Operations on sharded tensors work transparently (like JAX)
3. Reductions (sum, etc.) automatically gather from all shards
4. ModelParallel and DataParallel work correctly
5. No manual shard manipulation required in user code

