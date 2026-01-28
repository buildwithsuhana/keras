# Torch Distribution Library Implementation Plan

## Overview
Implement a complete distribution library for the PyTorch backend using DTensor, enabling model parallel and data parallel functionality similar to JAX.

## Files to Create/Modify

### 1. Create: `keras/src/backend/torch/distribution_lib.py`
**Purpose**: Core distribution utilities for PyTorch backend using DTensor

**Key Functions**:
- `list_devices(device_type)` - List available devices
- `get_device_count(device_type)` - Count available devices  
- `distribute_variable(value, layout)` - Create distributed variables
- `distribute_tensor(tensor, layout)` - Distribute tensors
- `_to_backend_mesh(device_mesh)` - Convert to torch DeviceMesh
- `_to_backend_layout(tensor_layout)` - Convert to DTensor Layout
- `initialize()`, `num_processes()`, `process_id()` - Process management

**DTensor Integration**:
```python
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)
```

### 2. Modify: `keras/src/backend/__init__.py`
**Changes**:
```python
# Change from:
elif backend() == "torch":
    from keras.src.backend.torch import *  # noqa: F403
    from keras.src.backend.torch.core import Variable as BackendVariable

    distribution_lib = None

# To:
elif backend() == "torch":
    from keras.src.backend.torch import *  # noqa: F403
    from keras.src.backend.torch.core import Variable as BackendVariable
    from keras.src.backend.torch import distribution_lib
```

### 3. Create: Path Adapter for Keras/PyTorch Path Conversion

**Location**: Inside `keras/src/backend/torch/distribution_lib.py`

**Purpose**: Convert Keras path format (`dense/kernel`) to PyTorch format (`dense.weight`)

```python
class TorchPathAdapter:
    """Adapt Keras path format to PyTorch format for regex matching.
    
    Keras uses / separators (e.g., 'dense/kernel')
    PyTorch uses . separators (e.g., 'dense.weight')
    """
    
    @staticmethod
    def keras_to_torch(keras_path: str) -> str:
        """Convert 'dense/kernel' to 'dense.*weight' for regex matching"""
        # Replace / with . for the base path
        torch_path = keras_path.replace('/', '.')
        
        # Handle special patterns for layer parameters
        # e.g., 'dense.*kernel' -> 'dense.*weight' or 'dense.*kernel'
        return torch_path
    
    @staticmethod  
    def torch_to_keras(torch_path: str) -> str:
        """Convert PyTorch path back to Keras format for matching"""
        # Convert dots back to slashes
        keras_path = torch_path.replace('.', '/')
        return keras_path
```

### 4. Model Parallelism Implementation Details

**Integration Points**:

#### A. Variable Sharding (`distribute_variable`)
```python
def distribute_variable(value, layout):
    """Distribute variable using DTensor based on layout."""
    if layout.device_mesh is None:
        return value
    
    # Get DTensor placements from layout
    placements = _get_placements_from_layout(layout)
    
    # Create DTensor from value
    dtensor = distribute_tensor(value, device_mesh, placements)
    return dtensor
```

#### B. Module Parallelization (`parallelize_module`)
For model parallelism, integrate with `torch.distributed.tensor.parallel`:

```python
def _apply_model_parallelism(module, layout_map, device_mesh):
    """Apply tensor parallelism to a module based on layout map."""
    for name, param in module.named_parameters():
        # Get the layout for this parameter
        keras_path = _get_keras_path_from_torch_name(name)
        layout = layout_map[keras_path]
        
        if layout is not None:
            # Determine parallel style from layout
            parallel_style = _infer_parallel_style(param, layout)
            
            # Apply parallelization
            if parallel_style == "colwise":
                return ColwiseParallel()
            elif parallel_style == "rowwise":
                return RowwiseParallel()
```

#### C. Layout to Placements Conversion
```python
def _get_placements_from_layout(layout):
    """Convert TensorLayout axes to DTensor placements."""
    placements = []
    for axis in layout.axes:
        if axis is None:
            placements.append(Replicate())
        elif axis == "batch":
            placements.append(Shard(0))  # Shard on batch dim
        elif axis == "model":
            placements.append(Shard(1))  # Shard on model dim
        else:
            placements.append(Replicate())
    return placements
```

### 5. Data Parallelism Implementation

**Dataset Distribution**:
```python
def distribute_dataset(dataset, layout, batch_dim_name):
    """Distribute dataset across devices."""
    # Use DistributedSampler for data parallelism
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=num_replicas,
        rank=rank,
        shuffle=True
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=global_batch_size // num_replicas
    )
    return dataloader
```

### 6. Test Files to Create

**Create**: `keras/src/backend/torch/distribution_lib_test.py`
- Test device listing and counting
- Test mesh and layout conversion
- Test variable distribution
- Test data distribution
- Test model parallelism with simple models

## Key Design Decisions

### DTensor vs Manual Sharding
- **Decision**: Use DTensor for all sharding operations
- **Rationale**: DTensor handles complex communication patterns automatically, similar to how JAX handles sharding

### Path Adapter Strategy
- **Approach**: Bidirectional conversion between Keras `/` and PyTorch `.` formats
- **Implementation**: Regex-based matching that works for both formats

### Parallel Style Inference
- Map Keras layout tuples to PyTorch parallel styles:
  - `(None, 'model')` → ColwiseParallel (shard output features)
  - `('model', None)` → RowwiseParallel (shard input features)

## Implementation Steps

1. Create basic `distribution_lib.py` with device listing
2. Add DTensor mesh/layout conversion functions
3. Implement `distribute_variable()` and `distribute_tensor()`
4. Add path adapter for Keras/PyTorch path conversion
5. Implement model parallelism with parallel styles
6. Add data parallelism with distributed sampler
7. Update `__init__.py` to import the new module
8. Write comprehensive tests
9. Integration testing with existing distribution tests

## Compatibility Notes

- Ensure backward compatibility with existing JAX distribution API
- Path adapter allows same regex patterns to work for both backends
- Model parallel behavior should be consistent across backends
- Data parallel should work seamlessly with PyTorch's native DDP

