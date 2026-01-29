# PyTorch Distribution Support Implementation

## TODO List

### Phase 1: Core Distribution Library
- [x] 1.1 Create `keras/src/backend/torch/distribution_lib.py` with DTensor support
  - [x] list_devices() - detect CPU/GPU/TPU devices
  - [x] get_device_count() - count available devices
  - [x] initialize() - setup for multi-process/multi-device
  - [x] distribute_tensor() - distribute tensors using DTensor
  - [x] distribute_variable() - create distributed variables
  - [x] _to_backend_mesh() - convert DeviceMesh to DTensor mesh
  - [x] _to_backend_layout() - convert TensorLayout to DTensor layout
  - [x] Path adapter for converting Keras `/` paths to PyTorch `.` paths

- [x] 1.2 Update `keras/src/backend/torch/__init__.py` to import distribution_lib

### Phase 2: Path Adapter
- [x] 2.1 Add path adapter utilities in `keras/src/distribution/distribution_lib.py`
- [x] 2.2 Modify `LayoutMap.__getitem__()` to check both Keras and PyTorch formats

### Phase 3: Debug Logging
- [x] 3.1 Add debug logging throughout distribution_lib.py
- [x] 3.2 Add debug logging in torch/distribution_lib.py

### Phase 4: Testing
- [x] 4.1 Create `keras/src/backend/torch/distribution_lib_test.py`
- [ ] 4.2 Run integration tests (pending user verification)

## Implementation Summary

### Files Created/Modified:
1. `keras/src/backend/torch/distribution_lib.py` - NEW (Core PyTorch distribution support)
2. `keras/src/backend/torch/__init__.py` - MODIFIED (Import distribution_lib)
3. `keras/src/backend/__init__.py` - MODIFIED (Import torch distribution_lib)
4. `keras/src/distribution/distribution_lib.py` - MODIFIED (Added path adapter, PyTorch support)
5. `keras/src/backend/torch/distribution_lib_test.py` - NEW (Tests)

### Key Features:
- Device detection for CPU, GPU, TPU
- DTensor integration for tensor distribution
- Path adapter for converting between Keras `/` and PyTorch `.` naming
- Debug logging support via `KERAS_DISTRIBUTION_DEBUG` env var
- Support for DataParallel and ModelParallel distributions

