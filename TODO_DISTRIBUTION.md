# PyTorch Distribution Support Implementation

## TODO List

### Phase 1: Core Distribution Library
- [ ] 1.1 Create `keras/src/backend/torch/distribution_lib.py` with DTensor support
  - [ ] list_devices() - detect CPU/GPU/TPU devices
  - [ ] get_device_count() - count available devices
  - [ ] initialize() - setup for multi-process/multi-device
  - [ ] distribute_tensor() - distribute tensors using DTensor
  - [ ] distribute_variable() - create distributed variables
  - [ ] _to_backend_mesh() - convert DeviceMesh to DTensor mesh
  - [ ] _to_backend_layout() - convert TensorLayout to DTensor layout
  - [ ] Path adapter for converting Keras `/` paths to PyTorch `.` paths

- [ ] 1.2 Update `keras/src/backend/torch/__init__.py` to import distribution_lib

### Phase 2: Path Adapter
- [ ] 2.1 Add path adapter utilities in `keras/src/distribution/distribution_lib.py`
- [ ] 2.2 Modify `LayoutMap.__getitem__()` to check both Keras and PyTorch formats

### Phase 3: Debug Logging
- [ ] 3.1 Add debug logging throughout distribution_lib.py
- [ ] 3.2 Add debug logging in torch/distribution_lib.py

### Phase 4: Testing
- [ ] 4.1 Create `keras/src/backend/torch/distribution_lib_test.py`
- [ ] 4.2 Add integration tests

## Implementation Details

### DTensor Integration
- Use PyTorch's experimental DTensor API
- Support CPU, GPU, and TPU devices
- Handle device mesh creation and tensor layout

### Path Conversion
- Keras uses: `dense/kernel` (forward slashes)
- PyTorch uses: `dense.weight` (dots)
- Need bidirectional conversion for regex matching

### Device Support
- CPU: `cpu:{id}`
- GPU: `cuda:{id}` or `cuda` for default
- TPU: `tpu:{id}` via XLA

