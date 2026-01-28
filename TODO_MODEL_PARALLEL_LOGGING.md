# Model Parallelism Logging Implementation Plan

## Overview
Add comprehensive logging to model parallelism functions in PyTorch to track:
- Distribution setup and initialization
- Tensor sharding and distribution operations
- Collective operations (all_gather, all_reduce, etc.)
- Optimizer updates across devices
- Data distribution

## Files to Modify

### 1. Create Logger Utility Module
- `keras/src/backend/torch/distribution_logger.py` - Centralized logging for distribution

### 2. Update Distribution Library
- `keras/src/backend/torch/distribution_lib.py` - Add logging to:
  - `initialize()` - Track distributed setup
  - `_get_current_rank()` - Log rank queries
  - `_get_current_device()` - Log device queries
  - `distribute_tensor()` - Log tensor distribution
  - `distribute_variable()` - Log variable distribution
  - `_shard_tensor()` - Log tensor sharding
  - `all_gather_variable()` - Log gather operations
  - `distribute_data_input()` - Log data distribution
  - `all_reduce()` - Log reduce operations
  - `all_gather()` - Log gather operations
  - `broadcast()` - Log broadcast operations

### 3. Update Optimizer Files
Add logging to `_parallel_update_step()` in:
- `keras/src/backend/torch/optimizers/torch_adam.py`
- `keras/src/backend/torch/optimizers/torch_sgd.py`
- `keras/src/backend/torch/optimizers/torch_adagrad.py`
- `keras/src/backend/torch/optimizers/torch_adadelta.py`
- `keras/src/backend/torch/optimizers/torch_adamax.py`
- `keras/src/backend/torch/optimizers/torch_nadam.py`
- `keras/src/backend/torch/optimizers/torch_lion.py`
- `keras/src/backend/torch/optimizers/torch_rmsprop.py`

### 4. Update Base Optimizer
- `keras/src/backend/torch/optimizers/torch_parallel_optimizer.py` - Add logging to base class

## Implementation Details

### Logging Format
```
[MODEL_PARALLEL:RANK:{rank}] {function_name}: {message}
[MODEL_PARALLEL:RANK:{rank}] {function_name}: {operation} - {details}
```

### Log Levels
- **INFO**: Major operations (initialization, distribution setup)
- **DEBUG**: Detailed operations (tensor shapes, sharding details)
- **WARNING**: Issues (missing DTensor, fallback operations)

### Configuration
- Environment variable: `KERAS_TORCH_MODEL_PARALLEL_LOG_LEVEL`
- Options: `DEBUG`, `INFO`, `WARNING`, `ERROR`
- Default: `INFO`

## Step-by-Step Implementation

### Step 1: Create Logger Module
- [x] Create `keras/src/backend/torch/distribution_logger.py`
- [x] Define logging functions
- [x] Add configuration handling

### Step 2: Update distribution_lib.py
- [x] Import logger module
- [x] Add logging to `initialize()`
- [x] Add logging to distribution functions
- [x] Add logging to collective operations

### Step 3: Update Optimizers
- [ ] Import logger in torch_parallel_optimizer.py
- [ ] Add logging to base class methods
- [ ] Update individual optimizer files

### Step 4: Testing
- [ ] Verify logging works in single device mode
- [ ] Verify logging works in distributed mode
- [ ] Test different log levels

## Expected Output Examples

### Initialization
```
[MODEL_PARALLEL:RANK:0] initialize: Starting distributed initialization
[MODEL_PARALLEL:RANK:0] initialize: Backend: nccl, World size: 2, Rank: 0
[MODEL_PARALLEL:RANK:0] initialize: Distributed initialization complete
```

### Tensor Distribution
```
[MODEL_PARALLEL:RANK:0] distribute_tensor: Sharding tensor shape (1024, 512) on axis 'model'
[MODEL_PARALLEL:RANK:0] distribute_tensor: Original shape: (1024, 512), Sharded shape: (256, 512)
[MODEL_PARALLEL:RANK:1] distribute_tensor: Sharding tensor shape (1024, 512) on axis 'model'
[MODEL_PARALLEL:RANK:1] distribute_tensor: Original shape: (1024, 512), Sharded shape: (256, 512)
```

### Optimizer Update
```
[MODEL_PARALLEL:RANK:0] _parallel_update_step: Starting optimizer update for 12 variables
[MODEL_PARALLEL:RANK:0] _parallel_update_step: Learning rate: 0.001, Momentum: 0.9
[MODEL_PARALLEL:RANK:0] _parallel_update_step: Update complete
```

## Notes
- Ensure logging doesn't impact performance significantly
- Use appropriate log levels to control verbosity
- Make logging configurable for production use
