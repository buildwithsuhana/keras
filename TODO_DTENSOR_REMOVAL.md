# TODO: Remove DTensor Dependency from Torch Distribution

## Goal
Modify the torch distribution system to use native PyTorch parallelism (torch.distributed, torch.chunk, torch.split) instead of DTensor for model parallel and data parallel support.

## Files to Modify
1. `keras/src/backend/torch/distribution_lib.py` - Core distribution logic
2. `keras/src/backend/torch/core.py` - Variable initialization with distribution
3. `keras/src/backend/torch/numpy.py` - NumPy operations with distribution
4. `keras/src/trainers/data_adapters/torch_data_loader_adapter.py` - Data loader distribution
5. `keras/src/trainers/epoch_iterator.py` - Epoch iterator distribution

## Implementation Steps

### Step 1: distribution_lib.py - Add Native Sharding Support
- [ ] Add `_get_shard_info()` helper function to extract sharding info from layout
- [ ] Add `_shard_tensor()` function for native tensor sharding
- [ ] Add `_get_sharded_slice()` helper for model parallel slicing
- [ ] Modify `distribute_variable()` to support native sharding (non-DTensor mode)
- [ ] Modify `distribute_tensor()` to support native sharding
- [ ] Modify `distribute_data_input()` for native data distribution
- [ ] Add `is_dtensor_available()` check for optional DTensor fallback
- [ ] Make DTensor optional (only if explicitly requested or available)

### Step 2: core.py - Variable Distribution
- [ ] Modify `Variable._initialize()` to use native sharding
- [ ] Add `_shard_variable_by_layout()` function
- [ ] Remove DTensor mode requirement in variable initialization
- [ ] Update `convert_to_tensor()` to handle sharded tensors

### Step 3: numpy.py - Operation Handling
- [ ] Remove automatic DTensor conversion in binary operations
- [ ] Use native PyTorch broadcasting for mismatched shapes
- [ ] Add `_ensure_matching_shape()` function for shape compatibility

### Step 4: torch_data_loader_adapter.py - Data Distribution
- [ ] Replace `_DTensorAwareDataLoader` with `_ShardAwareDataLoader`
- [ ] Simplify data sharding using torch.split/chunk
- [ ] Remove complex DTensor conversion logic

### Step 5: epoch_iterator.py - Pipeline Distribution
- [ ] Simplify `_distribute_data()` for native sharding
- [ ] Remove DTensor conversion from data pipeline

### Step 6: Testing
- [ ] Test data parallel on single GPU
- [ ] Test data parallel on multi-GPU (torch.distributed)
- [ ] Test model parallel (tensor sharding)
- [ ] Test combined model + data parallel

## Notes
- Keep DTensor as optional for backward compatibility
- Default to native PyTorch sharding
- Ensure torch.compile works with new implementation
- Remove "mixed torch.Tensor and DTensor" errors

## Progress
- [ ] Step 1: distribution_lib.py
- [ ] Step 2: core.py
- [ ] Step 3: numpy.py
- [ ] Step 4: torch_data_loader_adapter.py
- [ ] Step 5: epoch_iterator.py
- [ ] Step 6: Testing

