"""
Guide: Distributed Training with PyTorch Backend using DTensor

This guide explains how to use the distribution APIs with the PyTorch backend,
leveraging DTensor for efficient model and data parallelism.

## Introduction

The PyTorch backend now supports distribution strategies similar to JAX, using
PyTorch's DTensor API. This enables:

1. **Data Parallelism**: Distribute data across multiple devices/processes
2. **Model Parallelism**: Shard model weights across devices
3. **Combined Parallelism**: Mix data and model parallelism

## Setup

```python
import os
os.environ["KERAS_BACKEND"] = "torch"

import keras
from keras import layers
from keras.distribution import DeviceMesh, LayoutMap, ModelParallel, DataParallel
```

## Data Parallelism

Data parallelism replicates the model on all devices and splits the data:

```python
# List available devices
devices = keras.distribution.list_devices("gpu")
print(f"Available devices: {devices}")

# Create a data parallel distribution
distribution = DataParallel(devices=devices[:4])  # Use first 4 GPUs

# Or with explicit device mesh
device_mesh = DeviceMesh(
    shape=(4,),  # 4 devices
    axis_names=["batch"],
    devices=devices[:4]
)
distribution = DataParallel(device_mesh=device_mesh)

# Set as global distribution
keras.distribution.set_distribution(distribution)

# Create and train model as usual
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Dataset - use torch.utils.data.Dataset for auto-sharding
import torch
from torch.utils.data import TensorDataset

# Create torch dataset
x_train = torch.randn(1000, 784)
y_train = torch.randint(0, 10, (1000,))
dataset = TensorDataset(x_train, y_train)

# With distribution, the dataset will be automatically sharded
history = model.fit(dataset, epochs=5, batch_size=32)
```

## Model Parallelism

Model parallelism shards model weights across devices. This is useful for
large models that don't fit on a single device:

```python
# Create a device mesh with model parallelism
# Example: 2 data parallel devices, 4 model parallel devices
devices = keras.distribution.list_devices("gpu")
device_mesh = DeviceMesh(
    shape=(2, 4),  # 2 data, 4 model parallel
    axis_names=["batch", "model"],
    devices=devices[:8]
)

# Create a layout map for variable sharding
layout_map = LayoutMap(device_mesh)

# Define how variables should be sharded
# Keras path format: 'layer_name/parameter_name'
# E.g., 'dense/kernel' or 'dense.*kernel' with wildcard

# Shard the kernel of dense layers on the output dimension
layout_map['dense.*kernel'] = (None, 'model')  # Replicate input, shard output
layout_map['dense.*bias'] = ('model',)         # Shard bias

# For conv layers, shard on the output filters
layout_map['conv2d.*kernel'] = (None, None, None, 'model')
layout_map['conv2d.*bias'] = ('model',)

# Create the distribution
distribution = ModelParallel(
    layout_map=layout_map,
    batch_dim_name='batch'
)

# Set as global distribution
keras.distribution.set_distribution(distribution)

# Create model - variables will be automatically sharded
model = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, 3, activation='relu'),
    layers.Conv2D(64, 3, activation='relu'),
    layers.GlobalAveragePooling2D(),
    layers.Dense(10)
])

model.summary()
```

## Path Format Adaptation

Keras uses `/` separators for paths (e.g., `dense/kernel`), while PyTorch
uses `.` separators (e.g., `dense.weight`). The distribution library handles
this automatically:

```python
# Keras pattern (works for both backends)
layout_map['dense/kernel'] = (None, 'model')

# Wildcard patterns also work
layout_map['dense.*kernel'] = (None, 'model')

# For PyTorch, the library converts internally:
# 'dense/kernel' -> matches 'dense.weight'
# 'dense.*kernel' -> matches 'dense.weight', 'dense.kernel', etc.
```

## Manual Tensor Distribution

You can also manually distribute tensors:

```python
from keras.distribution import distribute_tensor, TensorLayout

# Create a layout
layout = TensorLayout(axes=('batch', None), device_mesh=device_mesh)

# Distribute a tensor
tensor = keras.backend.random.uniform((128, 784))
distributed_tensor = distribute_tensor(tensor, layout)
```

## Multi-Process Setup

For multi-process training, initialize the distribution:

```python
# On each process
import os
os.environ["KERAS_DISTRIBUTION_JOB_ADDRESSES"] = "10.0.0.1:1234,10.0.0.2:2345"
os.environ["KERAS_DISTRIBUTION_NUM_PROCESSES"] = "2"
os.environ["KERAS_DISTRIBUTION_PROCESS_ID"] = "0"  # Change for each process

# Initialize - this sets up the PyTorch process group
keras.distribution.initialize()

# Then create your distribution as usual
distribution = DataParallel()
keras.distribution.set_distribution(distribution)
```

## Complete Example: Model Parallel Training

```python
import os
os.environ["KERAS_BACKEND"] = "torch"

import keras
from keras import layers
from keras.distribution import DeviceMesh, LayoutMap, ModelParallel

# Setup device mesh for model parallelism
devices = keras.distribution.list_devices("gpu")
device_mesh = DeviceMesh(
    shape=(1, 4),  # 1 data dimension, 4 model dimensions
    axis_names=["batch", "model"],
    devices=devices[:4]
)

# Create layout map
layout_map = LayoutMap(device_mesh)

# Shard large layers
layout_map['.*kernel'] = (None, 'model')  # Shard output dimension
layout_map['.*bias'] = ('model',)

# Configure distribution
distribution = ModelParallel(
    layout_map=layout_map,
    batch_dim_name='batch'
)

with distribution.scope():
    # Create a large model
    model = keras.Sequential([
        layers.Input(shape=(512, 512, 3)),
        # These layers will be automatically sharded
        layers.Conv2D(1024, 3, activation='relu'),
        layers.Conv2D(1024, 3, activation='relu'),
        layers.Dense(4096, activation='relu'),
        layers.Dense(1000)
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy'
    )

# Train
print("Model created with distribution:", distribution)
print("Number of devices:", len(devices))
```

## Key Differences from JAX

While the API is similar to JAX, there are some differences:

1. **Eager Execution**: PyTorch/DTensor uses eager execution, unlike JAX's
   default lazy evaluation. Tensors are distributed immediately.

2. **Process Model**: PyTorch typically uses one process per device, while
   JAX can use one process with multiple devices.

3. **Dataloader Integration**: For PyTorch, use `torch.utils.data.Dataset`
   and `torch.utils.data.DataLoader` for best integration.

## Troubleshooting

### DTensor not available
If you see errors about DTensor, ensure you're using PyTorch 2.1+:
```bash
pip install torch --upgrade
```

### Device placement errors
Ensure all tensors are on the correct device:
```python
# Automatic device placement
tensor = tensor.cuda()  # For GPU
```

### Memory issues
Reduce batch size or shard more aggressively:
```python
# Increase model parallelism
device_mesh = DeviceMesh(shape=(1, 8), ...)  # More model shards
```

## API Reference

### keras.distribution.list_devices(device_type)
Returns available devices.

### keras.distribution.DeviceMesh
Cluster of computation devices.

### keras.distribution.TensorLayout
Layout specification for tensor distribution.

### keras.distribution.LayoutMap
Maps variable paths to layouts using regex.

### keras.distribution.DataParallel
Data parallelism distribution.

### keras.distribution.ModelParallel
Model parallelism distribution.

### keras.distribution.distribute_tensor(tensor, layout)
Manually distribute a tensor.

### keras.distribution.initialize(...)
Initialize multi-process distribution.
"""

