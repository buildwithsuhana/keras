# OPT-125M Distributed Training with Keras

This directory contains scripts for training OPT-125M from keras_hub using Keras distributed training APIs (DataParallel and ModelParallel) with PyTorch backend.

## Files

- `opt_distributed_kaggle.py` - Full-featured script with OPT-125M training and generation
- `opt_simple_test.py` - Simplified test script for quick verification

## Prerequisites

```bash
# Install required packages
pip install keras>=3.0
pip install keras-hub>=0.17.0
pip install torch torchvision
```

## Usage

### Single GPU

```bash
python opt_simple_test.py
```

or with full OPT-125M:

```bash
python opt_distributed_kaggle.py
```

### Multi-GPU (Kaggle with 2 T4 GPUs)

```bash
# Using torchrun
torchrun --nproc_per_node=2 opt_simple_test.py

# or for full version
torchrun --nproc_per_node=2 opt_distributed_kaggle.py
```

## What the Scripts Do

### DataParallel

- **How it works**: Replicates the model on each GPU. Each GPU processes different batches of data, then gradients are synchronized.
- **Use case**: Works well when you have multiple GPUs and want to increase batch size.
- **Sharding**: No model sharding; each GPU has a complete copy of the model.

```python
from keras.src.distribution import DataParallel, list_devices

dp = DataParallel(devices=devices, auto_shard_dataset=False)
with dp.scope():
    model = create_model()
```

### ModelParallel

- **How it works**: Splits the model across multiple GPUs. Each GPU holds a portion of the model weights.
- **Use case**: Required when the model is too large to fit on a single GPU.
- **Sharding**: Model weights are partitioned across devices using DTensor.

```python
from keras.src.distribution import ModelParallel, DeviceMesh, LayoutMap

mesh = DeviceMesh(
    shape=(1, len(devices)),
    axis_names=["batch", "model"],
    devices=devices
)

layout_map = LayoutMap(mesh)
layout_map[".*kernel"] = (None, "model")  # Shard kernels
layout_map[".*bias"] = ("model",)          # Shard biases

mp = ModelParallel(
    layout_map=layout_map,
    batch_dim_name="batch",
    auto_shard_dataset=False
)
```

## OPT-125M Architecture

The OPT-125M model has approximately 125 million parameters:

- **Vocabulary size**: 50,265 tokens
- **Hidden dimension**: 768
- **Number of layers**: 12
- **Number of attention heads**: 12
- **Feedforward dimension**: 3,072

### Weight Tensors to Shard

When using ModelParallel with OPT, the following tensors are good candidates for sharding:

1. **Token embedding**: `(50265, 768)` - Large vocabulary dimension
2. **Attention projections**: `(768, 768)` - Four projections (q, k, v, o)
3. **FFN layers**: `(768, 3072)` and `(3072, 768)` - Large feedforward weights
4. **Output layer**: `(768, 50265)` - Final projection to vocabulary

## Output Example

```
======================================================================
  OPT-125M WITH DATA PARALLEL
======================================================================
Using devices: ['cuda:0', 'cuda:1']
✓ DataParallel created: mesh_shape=(2,)
  Batch dimension: batch
✓ OPT-125M Backbone loaded: 125,237,760 parameters
  Vocabulary size: 50265
  Num layers: 12
  Hidden dim: 768
  Num heads: 12
✓ Full model created: 127,039,065 parameters
Training for 2 epochs...

Epoch 1/2: loss=5.2341 (time=123.4s)
Epoch 2/2: loss=4.8765 (time=120.1s)
✓ DataParallel Training Summary:
  - Total parameters: 127,039,065
  - Epochs completed: 2
  - Final loss: 4.8765
  - Total time: 243.5s
✓ DataParallel test PASSED
```

## Troubleshooting

### Common Issues

1. **"DTensor not available"**
   - Ensure PyTorch is compiled with distributed support
   - Check that `torch.cuda.is_available()` returns True

2. **"Mixed torch.Tensor and DTensor" error**
   - The scripts include fixes for this issue
   - Ensure inputs are converted to DTensors within distribution scope

3. **Out of memory on single GPU**
   - Reduce batch size
   - Use ModelParallel to shard weights across GPUs
   - Try mixed precision (not covered in these scripts)

4. **Keras hub model download fails**
   - Check internet connection
   - Try using a smaller model preset

### Environment Variables

```bash
# Enable debug logging
export KERAS_DISTRIBUTION_DEBUG=1

# Disable tokenizer parallelism
export TOKENIZERS_PARALLELISM=false

# Force PyTorch backend
export KERAS_BACKEND=torch
```

## Performance Tips

1. **DataParallel**: Increase batch size proportionally to GPU count
2. **ModelParallel**: Balance sharding across model dimensions
3. **Mixed Precision**: Consider adding `dtype="mixed_float16"` for memory efficiency
4. **Gradient Checkpointing**: Useful for large models (not implemented in these scripts)

## References

- [OPT: Open Pre-trained Transformer Language Models](https://arxiv.org/abs/2205.01068)
- [Keras Distributed Training Guide](https://keras.io/guides/distributed_training/)
- [PyTorch DTensor](https://pytorch.org/docs/stable/distributedTensor.html)

