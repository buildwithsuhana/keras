# TODO: Implement PyTorch Distribution Support with DTensor

## Phase 1: Create Torch Distribution Library
- [x] Create `keras/src/backend/torch/distribution_lib.py`
  - [x] Implement `list_devices()` for CPU/GPU/TPU detection
  - [x] Implement `get_device_count()` for device counting
  - [x] Implement `distribute_tensor()` using DTensor
  - [x] Implement `distribute_variable()` for variable distribution
  - [x] Implement `_to_backend_mesh()` conversion to DTensor Mesh
  - [x] Implement `_to_backend_layout()` conversion to DTensor Layout
  - [x] Implement multi-process initialization functions
  - [x] Add path separator adapter for Keras `/` to PyTorch `.` conversion

## Phase 2: Update Torch Backend Init
- [x] Update `keras/src/backend/torch/__init__.py`
  - [x] Import the new distribution_lib module

## Phase 3: Update High-Level Distribution Library
- [x] Update `keras/src/distribution/distribution_lib.py`
  - [x] Add path separator adapter in LayoutMap.__getitem__
  - [x] Make LayoutMap regex matching work with both `/` and `.` separators

## Phase 4: Create Tests
- [x] Add PyTorch-specific tests in `keras/src/distribution/distribution_lib_test.py`
  - [x] Test DataParallel distribution
  - [x] Test ModelParallel distribution
  - [x] Test path separator adapter
  - [x] Test device enumeration

## Phase 5: Create Example/Documentation
- [x] Create example demonstrating PyTorch distribution usage
- [ ] Update distributed_training_with_torch.py guide

## COMPLETED ✓
All main implementation tasks are complete. The PyTorch distribution
support with DTensor has been implemented, including:
- New distribution_lib.py for PyTorch backend
- Path separator adapter for Keras/PyTorch compatibility
- Updated high-level distribution library
- Comprehensive tests
- Example demonstrating the functionality

