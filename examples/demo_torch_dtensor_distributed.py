"""Example: Model and Data Parallelism with PyTorch DTensor

This example demonstrates how to use Keras distribution APIs with PyTorch
backend using DTensor for model and data parallelism.

Key features:
- Uses `torch.distributed.tensor` for DTensor support
- Handles Keras `/` vs PyTorch `.` separator conversion
- Supports CPU, GPU, and TPU devices
"""

import os

# Set backend before importing Keras
os.environ["KERAS_BACKEND"] = "torch"

import torch
import torch.nn as nn

import keras
from keras.src.distribution import DeviceMesh
from keras.src.distribution import LayoutMap
from keras.src.distribution import ModelParallel
from keras.src.distribution import set_distribution
from keras.src.distribution import TensorLayout


# ============================================
# Example 1: Basic Data Parallelism
# ============================================

def example_data_parallelism():
    """Example of data parallelism with PyTorch backend."""
    print("=" * 60)
    print("Example 1: Data Parallelism")
    print("=" * 60)

    from keras.src.backend.torch.distribution_lib import list_devices

    # List available devices
    devices = list_devices()
    print(f"Available devices: {devices}")

    from keras.src.distribution import DataParallel

    # Create data parallel distribution
    distribution = DataParallel()

    print(f"Distribution: {distribution}")
    print(f"Device mesh: {distribution.device_mesh}")


# ============================================
# Example 2: Model Parallelism with DTensor
# ============================================

def example_model_parallelism():
    """Example of model parallelism using DTensor."""
    print("\n" + "=" * 60)
    print("Example 2: Model Parallelism with DTensor")
    print("=" * 60)

    from keras.src.backend.torch.distribution_lib import (
        ModelParallel,
    )
    from keras.src.distribution import DeviceMesh
    from keras.src.distribution import LayoutMap
    from keras.src.distribution import TensorLayout

    # Create a device mesh with shape (data_parallel, model_parallel)
    # For example: 2 devices for data, 4 devices for model = 8 total devices
    devices = list(range(8))
    device_mesh = DeviceMesh(
        shape=(2, 4),
        axis_names=("batch", "model"),
        devices=[f"cuda:{i}" for i in devices] if torch.cuda.is_available()
                else [f"cpu:{i}" for i in devices],
    )

    # Create a layout map for model parallelism
    layout_map = LayoutMap(device_mesh)

    # Define sharding for dense layer weights
    # (None, 'model') means replicate first dim, shard second dim (output features)
    layout_map["dense.*kernel"] = TensorLayout([None, "model"])
    layout_map["dense.*bias"] = TensorLayout(["model"])

    # Define sharding for conv layer weights
    layout_map["conv2d.*kernel"] = TensorLayout(
        [None, None, None, "model"]
    )  # Shard output channels
    layout_map["conv2d.*bias"] = TensorLayout(["model"])

    # Create ModelParallel distribution
    distribution = ModelParallel(
        layout_map=layout_map,
        batch_dim_name="batch",
    )

    print(f"ModelParallel distribution: {distribution}")
    print(f"Device mesh shape: {device_mesh.shape}")
    print(f"Layout map: {layout_map._layout_map}")

    return distribution, device_mesh


# ============================================
# Example 3: Using ModelParallel Helper Class
# ============================================

def example_model_parallel_helper():
    """Example using ModelParallel helper class directly."""
    print("\n" + "=" * 60)
    print("Example 3: ModelParallel Helper Class")
    print("=" * 60)

    from keras.src.backend.torch.distribution_lib import ModelParallel

    # Create ModelParallel helper with device mesh shape
    # (2, 4) means 2 data parallel devices, 4 model parallel devices
    model_parallel = ModelParallel(
        device_mesh=(2, 4),
        layout_map={
            "dense.weight": (None, "model"),
            "dense.bias": ("model",),
            "conv.weight": (None, None, None, "model"),
        },
    )

    print(f"ModelParallel mesh: {model_parallel.mesh}")
    print(f"Mesh axis names: {model_parallel.mesh_axis_names}")

    # Create a simple PyTorch module
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.dense = nn.Linear(64, 128)

        def forward(self, x):
            return self.dense(x)

    model = SimpleModel()

    # Apply tensor parallelism to the module
    parallelized_model = model_parallel.parallelize_module(model)

    print(f"Original model: {model}")
    print(f"Parallelized model: {parallelized_model}")


# ============================================
# Example 4: Path Conversion Adapter
# ============================================

def example_path_conversion():
    """Example demonstrating path conversion between Keras and PyTorch."""
    print("\n" + "=" * 60)
    print("Example 4: Path Conversion Adapter")
    print("=" * 60)

    from keras.src.backend.torch.distribution_lib import (
        _convert_keras_path_to_torch,
        _convert_torch_path_to_keras,
    )

    # Keras uses '/' separators
    keras_paths = [
        "dense/kernel",
        "dense/bias",
        "conv2d_1/kernel",
        "my_model/layer_1/weight",
    ]

    # PyTorch uses '.' separators
    torch_paths = [
        "dense.weight",
        "dense.bias",
        "conv2d_1.weight",
        "my_model.layer_1.weight",
    ]

    print("Keras -> PyTorch conversion:")
    for keras_path in keras_paths:
        torch_path = _convert_keras_path_to_torch(keras_path)
        print(f"  {keras_path} -> {torch_path}")

    print("\nPyTorch -> Keras conversion:")
    for torch_path in torch_paths:
        keras_path = _convert_torch_path_to_keras(torch_path)
        print(f"  {torch_path} -> {keras_path}")


# ============================================
# Example 5: DTensor Distribution
# ============================================

def example_dtensor_distribution():
    """Example of distributing tensors with DTensor."""
    print("\n" + "=" * 60)
    print("Example 5: DTensor Distribution")
    print("=" * 60)

    from keras.src.backend.torch.distribution_lib import (
        distribute_tensor,
        _to_backend_layout,
    )
    from keras.src.distribution import DeviceMesh
    from keras.src.distribution import TensorLayout

    # Create a device mesh
    device_mesh = DeviceMesh(
        shape=(4,),
        axis_names=("data",),
        devices=[f"cpu:{i}" for i in range(4)],
    )

    # Create a layout that shards the first dimension
    layout = TensorLayout(axes=("data", None), device_mesh=device_mesh)

    # Create a tensor
    tensor = torch.randn(16, 8)

    # Distribute the tensor
    distributed = distribute_tensor(tensor, layout)

    print(f"Original tensor shape: {tensor.shape}")
    print(f"Distributed tensor: {distributed}")
    print(f"Layout: {layout}")


# ============================================
# Example 6: End-to-End Keras Model with Distribution
# ============================================

def example_keras_model_distribution():
    """Example of using distribution with Keras model."""
    print("\n" + "=" * 60)
    print("Example 6: Keras Model with Distribution")
    print("=" * 60)

    from keras.src.distribution import DataParallel
    from keras.src.distribution import DeviceMesh
    from keras.src.distribution import LayoutMap
    from keras.src.distribution import ModelParallel
    from keras.src.distribution import set_distribution
    from keras.src import layers
    from keras.src import Model

    # Create a simple model
    inputs = layers.Input(shape=(784,))
    x = layers.Dense(256, activation="relu")(inputs)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(10)(x)
    model = Model(inputs, outputs)

    # Show variable paths before distribution
    print("Variable paths before distribution:")
    for v in model.variables:
        print(f"  {v.path}")

    # Create a device mesh for model parallelism
    devices = list(range(4))
    device_mesh = DeviceMesh(
        shape=(2, 2),
        axis_names=("batch", "model"),
        devices=[f"cuda:{i}" for i in devices] if torch.cuda.is_available()
                else [f"cpu:{i}" for i in devices],
    )

    # Create layout map for model parallelism
    layout_map = LayoutMap(device_mesh)
    layout_map["dense.*kernel"] = (None, "model")  # Shard output dim
    layout_map["dense.*bias"] = ("model",)  # Shard bias

    # Create distribution
    distribution = ModelParallel(
        layout_map=layout_map,
        batch_dim_name="batch",
    )

    # Set as global distribution
    set_distribution(distribution)

    print(f"\nDistribution: {distribution}")

    # Get layouts for variables
    print("\nVariable layouts:")
    for v in model.variables:
        layout = distribution.get_variable_layout(v)
        print(f"  {v.path}: {layout}")


# ============================================
# Example 7: Parallel Style Inference
# ============================================

def example_parallel_style_inference():
    """Example of parallel style inference for different layer types."""
    print("\n" + "=" * 60)
    print("Example 7: Parallel Style Inference")
    print("=" * 60)

    from keras.src.backend.torch.distribution_lib import ModelParallel
    import torch.nn as nn

    model_parallel = ModelParallel(device_mesh=(2, 4))

    # Test different layer types
    layers_to_test = [
        (nn.Linear(64, 128), "weight", (None, "model"), "Linear - Colwise"),
        (nn.Linear(64, 128), "weight", ("model", None), "Linear - Rowwise"),
        (nn.Conv2d(3, 64, 3), "weight", (None, None, None, "model"), "Conv2d"),
        (nn.Embedding(1000, 256), "weight", ("model", None), "Embedding"),
    ]

    print("\nParallel style inference results:")
    for module, param_name, sharding_spec, description in layers_to_test:
        style = model_parallel._infer_parallel_style(
            module, param_name, sharding_spec
        )
        style_name = type(style).__name__ if style else "None"
        print(f"  {description}: {style_name}")


# ============================================
# Main
# ============================================

if __name__ == "__main__":
    print("PyTorch DTensor Distribution Examples")
    print("=" * 60)

    # Run all examples
    example_data_parallelism()
    example_model_parallelism()
    example_model_parallel_helper()
    example_path_conversion()
    example_dtensor_distribution()
    example_keras_model_distribution()
    example_parallel_style_inference()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)

