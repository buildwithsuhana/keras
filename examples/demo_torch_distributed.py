"""Example: Model Parallel and Data Parallel training with PyTorch backend.

This example demonstrates how to use the distribution APIs for PyTorch backend
similar to how it works with JAX backend.

Key features:
- Model parallelism: Shard large layers across devices
- Data parallelism: Replicate model across devices
- Path adapter: Keras regex patterns work with PyTorch parameter naming
"""

import os

# Set the backend to torch before importing keras
os.environ["KERAS_BACKEND"] = "torch"

import torch
import keras
from keras import layers, models
from keras.src.distribution import (
    DeviceMesh,
    TensorLayout,
    LayoutMap,
    ModelParallel,
    DataParallel,
    set_distribution,
)


def list_devices():
    """List available devices for PyTorch backend."""
    from keras.src.backend.torch import distribution_lib
    
    print("\n" + "=" * 50)
    print("Available Devices")
    print("=" * 50)
    
    # List all devices
    all_devices = distribution_lib.list_devices()
    print(f"All devices: {all_devices}")
    
    # List CPU devices
    cpu_devices = distribution_lib.list_devices("cpu")
    print(f"CPU devices: {cpu_devices}")
    
    # List GPU devices
    gpu_devices = distribution_lib.list_devices("gpu")
    print(f"GPU devices: {gpu_devices}")
    
    # Get device counts
    print(f"\nCPU count: {distribution_lib.get_device_count('cpu')}")
    print(f"GPU count: {distribution_lib.get_device_count('gpu')}")
    
    return all_devices


def demo_path_adapter():
    """Demonstrate the path adapter functionality."""
    from keras.src.backend.torch.distribution_lib import TorchPathAdapter
    
    print("\n" + "=" * 50)
    print("Path Adapter Demo")
    print("=" * 50)
    
    # Keras uses / separators
    keras_paths = [
        "dense/kernel",
        "dense/bias",
        "model/layer_1/weight",
        "model/layer_1/bias",
        "conv2d_1/kernel",
    ]
    
    # PyTorch uses . separators
    torch_paths = [
        "dense.weight",
        "dense.bias",
        "model.layer_1.weight",
        "model.layer_1.bias",
        "conv2d_1.kernel",
    ]
    
    print("\nKeras -> PyTorch conversion:")
    for k_path in keras_paths:
        t_path = TorchPathAdapter.keras_to_torch(k_path)
        print(f"  '{k_path}' -> '{t_path}'")
    
    print("\nPyTorch -> Keras conversion:")
    for t_path in torch_paths:
        k_path = TorchPathAdapter.torch_to_keras(t_path)
        print(f"  '{t_path}' -> '{k_path}'")
    
    print("\nPattern matching (Keras regex with PyTorch paths):")
    patterns = ["dense.*kernel", "dense.*bias", "conv2d.*"]
    for pattern in patterns:
        for t_path in torch_paths[:3]:
            match = TorchPathAdapter.match_pattern(pattern, t_path)
            print(f"  Pattern '{pattern}' matches '{t_path}': {match}")


def demo_data_parallel():
    """Demonstrate data parallelism with PyTorch backend."""
    from keras.src.backend.torch import distribution_lib
    
    print("\n" + "=" * 50)
    print("Data Parallel Demo")
    print("=" * 50)
    
    # Get available devices
    devices = distribution_lib.list_devices()
    print(f"\nUsing devices: {devices}")
    
    # Create DataParallel distribution
    distribution = DataParallel(devices=devices)
    print(f"\nCreated DataParallel distribution: {distribution}")
    
    # Set the distribution
    with distribution.scope():
        print("\nBuilding model inside distribution scope...")
        
        # Create a simple model
        model = models.Sequential([
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(10, activation='softmax'),
        ])
        
        # Print variable paths to show the naming
        print("\nModel variable paths:")
        for var in model.variables:
            print(f"  {var.path}")
    
    print("\nDataParallel demo completed!")


def demo_model_parallel():
    """Demonstrate model parallelism with PyTorch backend."""
    from keras.src.backend.torch import distribution_lib
    
    print("\n" + "=" * 50)
    print("Model Parallel Demo")
    print("=" * 50)
    
    # Get available devices
    devices = distribution_lib.list_devices()
    print(f"\nAvailable devices: {devices}")
    
    # Create a 2D mesh for model + data parallelism
    # Shape: (data_parallel, model_parallel)
    num_devices = len(devices)
    if num_devices >= 2:
        mesh_shape = (1, num_devices)  # All for model parallelism
        axis_names = ["batch", "model"]
    else:
        mesh_shape = (1,)
        axis_names = ["batch"]
    
    device_mesh = DeviceMesh(
        shape=mesh_shape,
        axis_names=axis_names,
        devices=devices,
    )
    print(f"\nCreated DeviceMesh: shape={mesh_shape}, axis_names={axis_names}")
    
    # Create layout map for model parallelism
    layout_map = LayoutMap(device_mesh)
    
    # Define sharding rules using Keras regex patterns
    # PyTorch path adapter will convert / to . automatically
    layout_map["dense.*kernel"] = TensorLayout([None, "model"], device_mesh)
    layout_map["dense.*bias"] = TensorLayout(["model"], device_mesh)
    layout_map["conv2d.*kernel"] = TensorLayout([None, None, None, "model"], device_mesh)
    layout_map["conv2d.*bias"] = TensorLayout(["model"], device_mesh)
    
    print("\nLayoutMap rules:")
    for key in layout_map:
        print(f"  {key}: {layout_map[key]}")
    
    # Create ModelParallel distribution
    distribution = ModelParallel(
        layout_map=layout_map,
        batch_dim_name="batch",
    )
    print(f"\nCreated ModelParallel distribution: {distribution}")
    
    # Set the distribution
    with distribution.scope():
        print("\nBuilding model inside ModelParallel scope...")
        
        # Create a model
        model = models.Sequential([
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(10, activation='softmax'),
        ])
        
        # Print variable paths
        print("\nModel variable paths:")
        for var in model.variables:
            print(f"  {var.path}")
            print(f"    shape: {var.shape}")
            print(f"    layout: {distribution.get_variable_layout(var)}")
    
    print("\nModelParallel demo completed!")


def demo_distribute_tensor():
    """Demonstrate tensor distribution."""
    from keras.src.backend.torch import distribution_lib
    from keras.src.distribution import DeviceMesh, TensorLayout
    
    print("\n" + "=" * 50)
    print("Distribute Tensor Demo")
    print("=" * 50)
    
    # Create a simple tensor
    tensor = torch.randn(8, 16, 32)
    print(f"\nOriginal tensor shape: {tensor.shape}")
    
    # Distribute with no sharding (replicate)
    layout_replicated = None
    result = distribution_lib.distribute_tensor(tensor, layout_replicated)
    print(f"Replicated tensor shape: {result.shape}")
    
    # Create a simple mesh
    devices = distribution_lib.list_devices()
    mesh = DeviceMesh(
        shape=(1,),
        axis_names=["batch"],
        devices=devices[:1] if devices else ["cpu:0"]
    )
    
    # Create a layout with first dimension sharded
    layout_sharded = TensorLayout([None], device_mesh=mesh)
    print(f"\nShard spec: {layout_sharded.axes}")
    
    print("\nDistribute tensor demo completed!")


def demo_multi_process():
    """Demonstrate multi-process distributed training setup."""
    from keras.src.backend.torch import distribution_lib
    
    print("\n" + "=" * 50)
    print("Multi-Process Setup Demo")
    print("=" * 50)
    
    # Simulate multi-process setup
    print("\nSetting up multi-process environment...")
    print("In a real scenario, you would:")
    print("1. Start multiple processes with different RANK values")
    print("2. Call distribution_lib.initialize() in each process")
    
    # Example initialization (would be called in each process)
    # distribution_lib.initialize(
    #     job_addresses="10.0.0.1:1234,10.0.0.2:2345",
    #     num_processes=2,
    #     process_id=0  # 0 or 1 for each process
    # )
    
    print(f"\nNumber of processes: {distribution_lib.num_processes()}")
    print(f"Current process ID: {distribution_lib.process_id()}")
    print(f"Is distributed: {distribution_lib.is_distributed()}")
    
    print("\nMulti-process demo completed!")


def run_all_demos():
    """Run all demonstration functions."""
    print("\n" + "#" * 60)
    print("# PyTorch Distribution Demo")
    print("#" * 60)
    
    # Run demos
    list_devices()
    demo_path_adapter()
    demo_distribute_tensor()
    demo_data_parallel()
    demo_model_parallel()
    demo_multi_process()
    
    print("\n" + "#" * 60)
    print("# All Demos Completed!")
    print("#" * 60)


if __name__ == "__main__":
    run_all_demos()

