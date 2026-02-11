"""Example: PyTorch Model and Data Parallel Training with Keras Distribution API.

This example demonstrates how to use the Keras distribution API with PyTorch backend
to perform both model parallelism and data parallelism training on CPU, GPU, and TPU.

Key Features:
- Model parallelism using PyTorch DTensor 
- Data parallelism using DistributedDataParallel
- Path conversion between Keras (dense/kernel) and PyTorch (dense.weight) formats
- Support for CPU, GPU, and TPU devices
"""

import os
# Set the backend before importing Keras
os.environ["KERAS_BACKEND"] = "torch"

import torch
import torch.nn as nn
import numpy as np

import keras
from keras.src.distribution import DeviceMesh, TensorLayout, LayoutMap, ModelParallel, DataParallel, set_distribution


def convert_keras_path_to_torch(keras_path):
    """Convert Keras layer parameter path to PyTorch format.
    
    Keras uses '/' separators (e.g., 'dense/kernel', 'dense_1/bias')
    PyTorch uses '.' separators (e.g., 'dense.weight', 'dense_1.bias')
    """
    # Basic conversion
    torch_path = keras_path.replace('/', '.')
    
    # Handle parameter name mapping
    replacements = [
        ('.kernel', '.weight'),
        ('.gamma', '.weight'),
        ('.beta', '.bias'),
        ('.moving_mean', '.running_mean'),
        ('.moving_var', '.running_var'),
    ]
    
    for old, new in replacements:
        if torch_path.endswith(old):
            torch_path = torch_path[:-len(old)] + new
            break
    
    return torch_path


def get_device_mesh_and_layout_map(device_type="gpu"):
    """Create DeviceMesh and LayoutMap for distribution.
    
    Args:
        device_type: One of "cpu", "gpu", or "tpu"
    
    Returns:
        device_mesh: DeviceMesh instance
        layout_map: LayoutMap instance for model parallelism
    """
    from keras.src.backend.torch import distribution_lib as torch_dist_lib
    
    # Get available devices
    devices = torch_dist_lib.list_devices(device_type)
    print(f"Available {device_type.upper()} devices: {devices}")
    
    num_devices = len(devices)
    
    if num_devices == 1:
        # Single device - no distribution needed
        return None, None
    
    # For model + data parallelism, we use a 2D mesh
    # First axis: data parallelism (batch splitting)
    # Second axis: model parallelism (weight sharding)
    if num_devices >= 4:
        # Use 2D mesh for combined parallelism
        data_parallel = 2
        model_parallel = num_devices // 2
        shape = (data_parallel, model_parallel)
        axis_names = ('batch', 'model')
    else:
        # Use 1D mesh for data parallelism only
        shape = (num_devices,)
        axis_names = ('batch',)
    
    # Create device mesh
    device_mesh = DeviceMesh(shape=shape, axis_names=axis_names, devices=devices)
    print(f"Created DeviceMesh: shape={shape}, axis_names={axis_names}")
    
    # Create layout map for model parallelism
    layout_map = LayoutMap(device_mesh)
    
    # Define sharding rules for common layers
    # Dense layer: shard the output dimension (model parallelism)
    layout_map['dense.*kernel'] = (None, 'model')
    layout_map['dense.*bias'] = ('model',)
    
    # Conv2D layer: shard the output filters
    layout_map['conv2d.*kernel'] = (None, None, None, 'model')
    layout_map['conv2d.*bias'] = ('model',)
    
    # BatchNormalization: replicate
    layout_map['batch_normalization.*gamma'] = ('model',)
    layout_map['batch_normalization.*beta'] = ('model',)
    
    print(f"Created LayoutMap with {len(layout_map)} rules")
    
    return device_mesh, layout_map


def create_model():
    """Create a simple model for demonstration."""
    inputs = keras.Input(shape=(28, 28, 1))
    x = keras.layers.Rescaling(1.0 / 255.0)(inputs)
    x = keras.layers.Conv2D(32, 3, activation='relu')(x)
    x = keras.layers.MaxPooling2D(2)(x)
    x = keras.layers.Conv2D(64, 3, activation='relu')(x)
    x = keras.layers.MaxPooling2D(2)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dense(10)(x)
    model = keras.Model(inputs, x)
    return model


def get_torch_device():
    """Get the appropriate PyTorch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        import torch_xla.core.xla_model as xm
        if xm.xla_device():
            return xm.xla_device()
    except ImportError:
        pass
    return torch.device("cpu")


def data_parallel_example():
    """Example of data parallel training with PyTorch backend."""
    print("\n=== Data Parallel Training Example ===")
    
    # Get device mesh and devices
    from keras.src.backend.torch import distribution_lib as torch_dist_lib
    devices = torch_dist_lib.list_devices("gpu")
    
    if len(devices) <= 1:
        print("Skipping data parallel - only 1 device available")
        return
    
    # Create data parallel distribution
    data_parallel = DataParallel(devices=devices)
    print(f"Created DataParallel: {data_parallel}")
    
    # Set distribution
    set_distribution(data_parallel)
    
    # Create and build model
    model = create_model()
    model.compile(
        optimizer='adam',
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    print("Model created with DataParallel distribution")
    print(f"Device mesh: {data_parallel.device_mesh}")
    print(f"Batch dim name: {data_parallel.batch_dim_name}")
    
    return model


def model_parallel_example():
    """Example of model parallel training with PyTorch backend."""
    print("\n=== Model Parallel Training Example ===")
    
    # Get device mesh and layout map
    device_mesh, layout_map = get_device_mesh_and_layout_map("gpu")
    
    if device_mesh is None:
        print("Skipping model parallel - only 1 device available")
        return
    
    # Create model parallel distribution
    model_parallel = ModelParallel(
        layout_map=layout_map,
        batch_dim_name='batch'
    )
    print(f"Created ModelParallel: {model_parallel}")
    
    # Set distribution
    set_distribution(model_parallel)
    
    # Create and build model
    model = create_model()
    model.compile(
        optimizer='adam',
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    print("Model created with ModelParallel distribution")
    print(f"Device mesh: {model_parallel.device_mesh}")
    print(f"Layout map rules: {len(model_parallel._layout_map)}")
    
    # Print variable paths and their layouts
    print("\nVariable paths and layouts:")
    for var in model.trainable_variables:
        layout = model_parallel.get_variable_layout(var)
        print(f"  {var.path}: {layout}")
    
    return model


def path_conversion_example():
    """Example demonstrating path conversion between Keras and PyTorch formats."""
    print("\n=== Path Conversion Example ===")
    
    from keras.src.backend.torch.distribution_lib import (
        _convert_keras_path_to_torch,
        _convert_torch_path_to_keras
    )
    
    # Test Keras to PyTorch conversion
    keras_paths = [
        "dense/kernel",
        "dense_1/bias",
        "conv2d/kernel",
        "batch_normalization/gamma",
        "batch_normalization/moving_mean"
    ]
    
    print("Keras Path -> PyTorch Path:")
    for keras_path in keras_paths:
        torch_path = _convert_keras_path_to_torch(keras_path)
        print(f"  {keras_path:30s} -> {torch_path}")
    
    # Test PyTorch to Keras conversion
    torch_paths = [
        "dense.weight",
        "dense_1.bias",
        "conv2d.weight",
        "batch_normalization.weight",
        "batch_normalization.running_mean"
    ]
    
    print("\nPyTorch Path -> Keras Path:")
    for torch_path in torch_paths:
        keras_path = _convert_torch_path_to_keras(torch_path)
        print(f"  {torch_path:30s} -> {keras_path}")


def combined_parallel_example():
    """Example of combined data and model parallelism."""
    print("\n=== Combined Data + Model Parallel Example ===")
    
    from keras.src.backend.torch import distribution_lib as torch_dist_lib
    devices = torch_dist_lib.list_devices("gpu")
    num_devices = len(devices)
    
    if num_devices < 4:
        print("Skipping combined parallel - need at least 4 devices")
        return
    
    # Create combined distribution
    device_mesh, layout_map = get_device_mesh_and_layout_map("gpu")
    
    if device_mesh is None:
        print("Skipping - only 1 device available")
        return
    
    # Create combined distribution
    distribution = ModelParallel(
        layout_map=layout_map,
        batch_dim_name='batch'
    )
    
    set_distribution(distribution)
    
    # Create model
    model = create_model()
    model.compile(
        optimizer='adam',
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    print(f"Created combined Data + Model parallel distribution")
    print(f"Device mesh shape: {distribution.device_mesh.shape}")
    print(f"Total devices: {num_devices}")
    
    # Show variable layouts
    print("\nVariable layouts:")
    for var in model.trainable_variables:
        layout = distribution.get_variable_layout(var)
        if layout is not None:
            print(f"  {var.path}: axes={layout.axes}")


def tpu_example():
    """Example of TPU distribution (when available)."""
    print("\n=== TPU Distribution Example ===")
    
    try:
        import torch_xla.core.xla_model as xm
        
        # Check if TPU is available
        devices = keras.distribution.list_devices("tpu")
        if not devices:
            print("No TPU devices available")
            return
        
        print(f"TPU devices found: {devices}")
        
        # Create distribution for TPU
        device_mesh, layout_map = get_device_mesh_and_layout_map("tpu")
        
        if device_mesh is not None:
            distribution = ModelParallel(
                layout_map=layout_map,
                batch_dim_name='batch'
            )
            
            set_distribution(distribution)
            print("TPU distribution configured")
        
    except ImportError:
        print("PyTorch XLA not available. Install with: pip install torch-xla")


def main():
    """Run all examples."""
    print("PyTorch Distribution API Examples")
    print("=" * 50)
    
    # Check backend
    print(f"Keras backend: {keras.config.backend()}")
    
    # Run examples
    path_conversion_example()
    
    try:
        data_parallel_example()
    except Exception as e:
        print(f"Data parallel example failed: {e}")
    
    try:
        model_parallel_example()
    except Exception as e:
        print(f"Model parallel example failed: {e}")
    
    try:
        combined_parallel_example()
    except Exception as e:
        print(f"Combined parallel example failed: {e}")
    
    try:
        tpu_example()
    except Exception as e:
        print(f"TPU example failed: {e}")
    
    print("\n" + "=" * 50)
    print("Examples completed!")


if __name__ == "__main__":
    main()
