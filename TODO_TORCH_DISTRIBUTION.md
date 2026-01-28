# Torch Distribution Implementation - TODO

## Completed Tasks

### 1. Core Distribution Library
- [x] Created `keras/src/backend/torch/distribution_lib.py`
- [x] Implemented `list_devices()` for CPU/GPU/TPU
- [x] Implemented `get_device_count()` for all device types
- [x] Implemented `distribute_variable()` with DTensor support
- [x] Implemented `distribute_tensor()` with layout support
- [x] Implemented `_to_backend_mesh()` for DeviceMesh conversion
- [x] Implemented `_to_backend_layout()` for TensorLayout conversion
- [x] Implemented `initialize()` for multi-process setup
- [x] Implemented helper functions: `num_processes()`, `process_id()`, `is_distributed()`
- [x] Implemented collective operations: `all_reduce()`, `broadcast()`, `scatter_tensor()`, `gather_tensor()`
- [x] Implemented `replicate_model()` for DDP support

### 2. Path Adapter
- [x] Created `TorchPathAdapter` class
- [x] Implemented `keras_to_torch()` conversion (`/` -> `.`)
- [x] Implemented `torch_to_keras()` conversion (`.` -> `/`)
- [x] Implemented `match_pattern()` for regex matching with PyTorch paths
- [x] Implemented caching for performance

### 3. Variable Integration
- [x] Updated `keras/src/backend/torch/core.py`
- [x] Added `__init__` with layout parameter to `Variable` class
- [x] Added `_initialize_layout()` method
- [x] Updated `_initialize()` to apply distribution
- [x] Updated `_direct_assign()` to handle distribution

### 4. Layer Integration
- [x] Updated `keras/src/backend/torch/layer.py`
- [x] Updated `_track_variables()` to track both Keras and PyTorch paths
- [x] Updated `_post_track_variable()` with path adapter support
- [x] Updated `_post_untrack_variable()` with path adapter support

### 5. Module Exports
- [x] Updated `keras/src/backend/torch/__init__.py` to export distribution_lib

### 6. LayoutMap Integration
- [x] Updated `keras/src/distribution/distribution_lib.py`
- [x] Added `_get_path_adapter()` method to LayoutMap
- [x] Updated `__getitem__()` to support PyTorch path lookup

### 7. Tests
- [x] Created `keras/src/backend/torch/distribution_test.py`
- [x] Tests for device listing and counting
- [x] Tests for path adapter conversion and matching
- [x] Tests for distribute_variable and distribute_tensor
- [x] Tests for distribution initialization
- [x] Tests for collective operations

### 8. Examples
- [x] Created `examples/demo_torch_distributed.py`
- [x] Demo for device listing
- [x] Demo for path adapter
- [x] Demo for data parallelism
- [x] Demo for model parallelism
- [x] Demo for tensor distribution
- [x] Demo for multi-process setup

## Remaining Tasks

### 9. Documentation
- [ ] Add docstrings to all public functions
- [ ] Create API documentation
- [ ] Update main README with PyTorch distribution examples

### 10. Integration Tests
- [ ] Test with actual multi-GPU setup
- [ ] Test with DTensor (if available)
- [ ] Test with TPU (if available)

### 11. Bug Fixes and Refinements
- [ ] Handle edge cases in path adapter
- [ ] Optimize performance for large models
- [ ] Fix any issues found during testing

## Usage Instructions

### Basic Usage

```python
import os
os.environ["KERAS_BACKEND"] = "torch"

import keras
from keras import layers
from keras.src.distribution import (
    DeviceMesh,
    TensorLayout,
    LayoutMap,
    ModelParallel,
    DataParallel,
    set_distribution,
)

# Data Parallel Example
devices = keras.distribution.list_devices()  # or ["cpu:0"] for CPU only
distribution = DataParallel(devices=devices)
set_distribution(distribution)

model = keras.Sequential([...])
model.compile()
model.fit(data)

# Model Parallel Example
devices = list_devices()
mesh = DeviceMesh(shape=(2, 2), axis_names=["batch", "model"], devices=devices)
layout_map = LayoutMap(mesh)
layout_map["dense.*kernel"] = TensorLayout([None, "model"], mesh)
layout_map["dense.*bias"] = TensorLayout(["model"], mesh)

distribution = ModelParallel(layout_map=layout_map, batch_dim_name="batch")
set_distribution(distribution)

model = keras.Sequential([...])
model.compile()
model.fit(data)
```

### Multi-Process Usage

```python
# In each process
from keras.src.backend.torch import distribution_lib

distribution_lib.initialize(
    job_addresses="10.0.0.1:1234,10.0.0.2:2345",
    num_processes=2,
    process_id=0  # or 1 in the other process
)
```

## Key Features

1. **Path Adapter**: Keras regex patterns like `dense.*kernel` work seamlessly with PyTorch parameter names like `dense.weight`
2. **Device Support**: CPU, GPU, and TPU support
3. **DTensor Integration**: Full DTensor support when available (PyTorch 2.1+)
4. **Fallback Mode**: Works even without DTensor using standard PyTorch
5. **Print Statements**: Debugging information for understanding distribution behavior

