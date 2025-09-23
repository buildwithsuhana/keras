# Keras Tensor Parallel Implementation

This is a port of the PyTorch `tensor_parallel` library to Keras 3.0. It provides tensor parallelism capabilities for Keras models, allowing you to distribute model parameters across multiple devices for parallel computation.

## Features

- **Automatic Sharding**: Automatically detects and shards Dense, Embedding, Conv2D, LSTM, and MultiHeadAttention layers
- **Multi-Device Support**: Works with CPU and GPU devices
- **Keras 3.0 Compatible**: Built for the latest version of Keras
- **PyTorch Backend**: Leverages PyTorch for tensor operations and device management

## Architecture

The implementation consists of several key components:

### Core Classes

- **`TensorParallelKeras`**: Main wrapper class that handles model parallelization
- **`ConfigKeras`**: Configuration management for tensor parallel operations
- **`StateActionKeras`**: Handles parameter splitting, gathering, and reduction

### Communication Operations

- **`AllReduceKeras`**: For gradient synchronization
- **`AllGatherKeras`**: For output collection
- **`BroadcastKeras`**: For input distribution

### Sharding

- **`SplitKeras`**: Splits tensors along specified dimensions
- **`GatherKeras`**: Gathers tensors from multiple devices
- **`SumKeras`**: Reduces tensors by summing

## Usage

### Basic Usage

```python
import keras
from keras import layers, Model
from tensor_parallel_keras import TensorParallelKeras

# Create a simple model
inputs = keras.Input(shape=(100,))
x = layers.Dense(200, activation='relu')(inputs)
x = layers.Dense(10, activation='softmax')(x)
model = Model(inputs=inputs, outputs=x)

# Create tensor parallel model
tp_model = TensorParallelKeras(
    model,
    device_ids=["cpu", "cpu"],  # Use 2 CPU devices
    sharded=True
)

# Use the model
output = tp_model(input_data)
```

### Advanced Configuration

```python
# Custom device configuration
tp_model = TensorParallelKeras(
    model,
    device_ids=["gpu:0", "gpu:1"],  # Use 2 GPU devices
    output_device="gpu:0",           # Output on first GPU
    distributed=True,                 # Enable distributed operations
    sharded=True                     # Enable parameter sharding
)
```

## Supported Layer Types

### Fully Supported
- **Dense**: Split along output dimension
- **Embedding**: Split along embedding dimension
- **Conv2D/Conv1D/Conv3D**: Split along output channels
- **LSTM**: Split along hidden size dimension
- **MultiHeadAttention**: Split query, key, value projections

### Partially Supported
- **LayerNorm**: No sharding (parameters shared)
- **ReLU/Activation**: No sharding (no parameters)
- **Dropout**: No sharding (no parameters)

## Device Support

- **CPU**: Full support with cross-device operations
- **GPU**: Full support with CUDA operations (when available)
- **Mixed**: Support for mixed CPU/GPU setups

## Limitations

1. **PyTorch Dependency**: Currently requires PyTorch for tensor operations
2. **Limited Communication**: Cross-device communication is simplified
3. **No LoRA Support**: LoRA adapters are not supported in this version
4. **Training Only**: Focused on training scenarios, inference optimization may vary

## Testing

Run the basic tests:

```bash
cd tests
python test_keras_basic.py
```

## Future Improvements

1. **Native Keras Operations**: Replace PyTorch dependencies with native Keras operations
2. **Advanced Communication**: Implement more sophisticated cross-device communication
3. **Performance Optimization**: Optimize for inference and training performance
4. **More Layer Types**: Support for additional Keras layer types
5. **Distributed Training**: Better integration with Keras distributed training

## Contributing

This is a work-in-progress port. Contributions are welcome to improve:

- Layer support
- Performance optimization
- Testing coverage
- Documentation

## License

Same license as the original PyTorch tensor_parallel library. 