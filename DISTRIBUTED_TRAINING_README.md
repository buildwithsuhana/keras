 a# PyTorch Distributed Training Verification Scripts

This directory contains comprehensive verification scripts for testing and validating PyTorch distributed training with Keras backend.

## Scripts Overview

### 1. `verify_torch_distribution.py`
Comprehensive verification class with multiple test methods:
- Device detection and enumeration
- DeviceMesh creation and configuration
- DataParallel functionality testing
- ModelParallel sharding verification
- Gradient synchronization testing

**Usage:**
```bash
# Single process
python verify_torch_distribution.py

# Multi-GPU distributed
torchrun --nproc_per_node=2 verify_torch_distribution.py
```

### 2. `test_torch_distributed.py`
Enhanced training script with detailed logging:
- Process rank and device information
- Distribution configuration display
- Training progress with detailed metrics
- Model sharding verification

**Usage:**
```bash
# Single process
python test_torch_distributed.py

# Multi-GPU
torchrun --nproc_per_node=2 test_torch_distributed.py
```

### 3. `launch_distributed_test.py`
Simple launcher script for quick verification:
- Automated test execution
- Configurable test types (dp, mp, all)
- Number of epochs configurable
- Verbose logging support

**Usage:**
```bash
# Single process (for debugging)
python launch_distributed_test.py --single-process

# Multi-GPU distributed
torchrun --nproc_per_node=2 launch_distributed_test.py

# Run only DataParallel test
torchrun --nproc_per_node=2 launch_distributed_test.py --test=dp

# Run only ModelParallel test with 5 epochs
torchrun --nproc_per_node=2 launch_distributed_test.py --test=mp --epochs=5
```

## Environment Variables

The following environment variables control behavior:

- `KERAS_BACKEND`: Set to `"torch"` for PyTorch backend
- `KERAS_DISTRIBUTION_DEBUG`: Set to `"1"` for debug logging
- `KERAS_DISTRIBUTION_JOB_ADDRESSES`: Comma-separated IP addresses for cluster
- `KERAS_DISTRIBUTION_NUM_PROCESSES`: Number of processes in the cluster
- `KERAS_DISTRIBUTION_PROCESS_ID`: Current process ID (0 to N-1)

## Expected Output

### DataParallel Test Output
```
============================================================================
TEST: DataParallel
============================================================================
Devices: ['cuda:0', 'cuda:1']
Number of devices: 2
DataParallel created: shape=(2,), axis_names=['batch']
Model parameters: 10,890
Data shape: (32, 64) -> (32, 10)
Training...
  Epoch 1/2: loss=0.4567
  Epoch 2/2: loss=0.4234
✓ DataParallel test passed
```

### ModelParallel Test Output
```
============================================================================
TEST: ModelParallel
============================================================================
Devices: ['cuda:0', 'cuda:1']
DeviceMesh: shape=(1, 2), axes=['batch', 'model']
LayoutMap configured:
  - dense.*kernel: axes=(None, 'model')
  - dense.*bias: axes=('model',)
ModelParallel created: batch_dim_name=batch
Model parameters: 98,506
Data shape: (32, 128) -> (32, 10)
Training...
  Epoch 1/2: loss=0.5123
  Epoch 2/2: loss=0.4891
✓ ModelParallel test passed
```

## Verification Checklist

The verification scripts check the following:

### Device Detection
- ✓ GPU enumeration
- ✓ Device properties (memory, compute capability)
- ✓ Keras device listing

### DataParallel
- ✓ DeviceMesh creation
- ✓ Model instantiation
- ✓ Forward pass
- ✓ Loss computation
- ✓ Training step
- ✓ Weight updates
- ✓ Multiple epoch training

### ModelParallel
- ✓ 2D DeviceMesh creation
- ✓ LayoutMap configuration
- ✓ Tensor sharding
- ✓ Model weight distribution
- ✓ Forward pass with sharded weights
- ✓ Training with sharded model
- ✓ Gradient computation

### Gradient Synchronization
- ✓ Gradient computation
- ✓ Gradient norms
- ✓ Parameter gradients

## Troubleshooting

### No GPU Detected
```
⚠ No GPU detected, using CPU
```
- Check CUDA installation: `nvidia-smi`
- Verify PyTorch CUDA support: `python -c "import torch; print(torch.cuda.is_available())"`

### Distributed Initialization Failed
```
Error: nccl backend initialization failed
```
- Ensure NCCL is available (Linux only)
- Check network connectivity for multi-node training
- Try gloo backend for CPU-only training

### Import Errors
```
ModuleNotFoundError: No module named 'keras'
```
- Ensure you're in the keras source directory
- Install keras: `pip install -e .`
- Set `KERAS_BACKEND` before importing keras

## Integration with Your Code

Use the `DistributionVerifier` class in your own code:

```python
from verify_torch_distribution import DistributionVerifier

# Create verifier
verifier = DistributionVerifier(verbose=True)

# Run all tests
results = verifier.run_all_tests()

# Print summary
verifier.print_summary_report(results)
```

## Running with Your Training Script

1. Copy relevant parts from `verify_torch_distribution.py`
2. Add logging calls to your training loop
3. Use `DistributedLogger` for multi-process safe logging
4. Verify gradient synchronization
5. Check weight distribution for ModelParallel
