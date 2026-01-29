"""
Example: PyTorch Distribution with DTensor

This example demonstrates how to use the new Keras distribution APIs
with PyTorch backend using DTensor for efficient distributed training.

This implementation supports:
- Data Parallelism: Distribute data across devices while replicating model
- Model Parallelism: Shard model weights across devices
- Path separator adapter: Works with both Keras (dense/kernel) and
  PyTorch (dense.weight) naming conventions
"""

import os

# Set backend to torch
os.environ["KERAS_BACKEND"] = "torch"

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)

import keras
from keras import layers
import numpy as np

# Import the new distribution library
from keras.src.distribution import (
    DeviceMesh,
    TensorLayout,
    DataParallel,
    ModelParallel,
    LayoutMap,
    list_devices,
    set_distribution,
)
from keras.src.backend.torch.distribution_lib import (
    convert_keras_path_to_pytorch,
    convert_pytorch_path_to_keras,
    PathSeparatorAdapter,
)


def get_model():
    """Create a simple model for demonstration."""
    inputs = keras.Input(shape=(28, 28, 1))
    x = layers.Rescaling(1.0 / 255.0)(inputs)
    x = layers.Conv2D(
        filters=32, kernel_size=3, padding="same", activation="relu"
    )(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(
        filters=64, kernel_size=3, padding="same", activation="relu"
    )(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10)(x)
    model = keras.Model(inputs, outputs)
    return model


def get_dataset():
    """Get a simple dataset for testing."""
    (x_train, y_train), _ = keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, -1)
    # Use a small subset for quick testing
    x_train = x_train[:1000]
    y_train = y_train[:1000]
    return x_train, y_train


def demonstrate_data_parallel():
    """Demonstrate data parallel distribution."""
    print("\n" + "=" * 60)
    print("Data Parallel Distribution Example")
    print("=" * 60)

    # List available devices
    devices = list_devices()
    print(f"\nAvailable devices: {devices}")

    if len(devices) < 2:
        print("Note: Multiple devices needed for true parallelism.")
        print("Demonstrating with single device...")

    # Create a device mesh for data parallelism
    device_mesh = DeviceMesh(
        shape=(len(devices),),
        axis_names=["batch"],
        devices=devices,
    )

    # Create data parallel distribution
    distribution = DataParallel(device_mesh=device_mesh)

    # Set the distribution
    set_distribution(distribution)

    # Create and build model
    model = get_model()
    x_train, y_train = get_dataset()

    # Build the model
    model.build(input_shape=(None, 28, 28, 1))

    print(f"\nModel created with {len(model.trainable_variables)} variables")
    for v in model.trainable_variables:
        print(f"  {v.path}: {v.shape}")

    # Demonstrate that layout works
    for v in model.trainable_variables[:3]:
        layout = distribution.get_variable_layout(v)
        print(f"\nVariable: {v.path}")
        print(f"  Layout axes: {layout.axes}")
        print(f"  Device mesh: {layout.device_mesh}")

    print("\nData Parallel distribution setup complete!")


def demonstrate_model_parallel():
    """Demonstrate model parallel distribution."""
    print("\n" + "=" * 60)
    print("Model Parallel Distribution Example")
    print("=" * 60)

    # List available devices
    devices = list_devices()
    print(f"\nAvailable devices: {devices}")

    if len(devices) < 2:
        print("Note: Multiple devices needed for true model parallelism.")
        print("Demonstrating with single device...")

    # Create a 2D device mesh for model + data parallelism
    # Shape: (data_parallel_dim, model_parallel_dim)
    num_devices = len(devices)
    if num_devices >= 4:
        mesh_shape = (2, 2)  # 2 data parallel x 2 model parallel
    elif num_devices >= 2:
        mesh_shape = (1, 2)  # 1 data parallel x 2 model parallel
    else:
        mesh_shape = (1, 1)

    device_mesh = DeviceMesh(
        shape=mesh_shape,
        axis_names=["data", "model"],
        devices=devices[: num_devices],
    )

    # Create a layout map for model parallelism
    layout_map = LayoutMap(device_mesh)

    # Define sharding for different layer types
    # Keras path format (dense/kernel) is automatically converted to
    # PyTorch format (dense.weight) by the adapter
    layout_map["dense.*kernel"] = TensorLayout([None, "model"], device_mesh)
    layout_map["dense.*bias"] = TensorLayout(["model"], device_mesh)
    layout_map["conv2d.*kernel"] = TensorLayout(
        [None, None, None, "model"], device_mesh
    )
    layout_map["conv2d.*bias"] = TensorLayout(["model"], device_mesh)

    # Create model parallel distribution
    distribution = ModelParallel(
        layout_map=layout_map, batch_dim_name="data"
    )

    # Set the distribution
    set_distribution(distribution)

    # Create and build model
    model = get_model()
    x_train, y_train = get_dataset()

    # Build the model
    model.build(input_shape=(None, 28, 28, 1))

    print(f"\nModel created with {len(model.trainable_variables)} variables")

    # Check variable layouts
    print("\nVariable layouts:")
    for v in model.trainable_variables:
        layout = distribution.get_variable_layout(v)
        print(f"  {v.path}: axes={layout.axes}")

    print("\nModel Parallel distribution setup complete!")


def demonstrate_path_separator_adapter():
    """Demonstrate the path separator adapter functionality."""
    print("\n" + "=" * 60)
    print("Path Separator Adapter Demo")
    print("=" * 60)

    # Test Keras to PyTorch conversion
    print("\nKeras path -> PyTorch path:")
    test_cases = [
        ("dense/kernel", "dense.weight"),
        ("dense/bias", "dense.bias"),
        ("conv2d/kernel", "conv2d.weight"),
        ("conv2d/bias", "conv2d.bias"),
        ("layer_norm/gamma", "layer_norm.weight"),
        ("layer_norm/beta", "layer_norm.bias"),
    ]

    for keras_path, expected_pytorch in test_cases:
        result = convert_keras_path_to_pytorch(keras_path)
        status = "✓" if result == expected_pytorch else "✗"
        print(f"  {status} {keras_path} -> {result}")

    # Test PyTorch to Keras conversion
    print("\nPyTorch path -> Keras path:")
    test_cases = [
        ("dense.weight", "dense/kernel"),
        ("dense.bias", "dense/bias"),
        ("conv2d.weight", "conv2d/kernel"),
        ("conv2d.bias", "conv2d/bias"),
    ]

    for pytorch_path, expected_keras in test_cases:
        result = convert_pytorch_path_to_keras(pytorch_path)
        status = "✓" if result == expected_keras else "✗"
        print(f"  {status} {pytorch_path} -> {result}")

    # Test pattern matching with both formats
    print("\nPattern matching with both formats:")
    pattern = r"dense.*kernel"
    test_paths = ["dense/kernel", "dense.weight"]
    for path in test_paths:
        match = PathSeparatorAdapter.match_pattern(pattern, path)
        status = "✓" if match else "✗"
        print(f"  {status} Pattern '{pattern}' matches '{path}': {match}")


def demonstrate_multi_gpu_training():
    """Demonstrate full multi-GPU training with distribution."""
    print("\n" + "=" * 60)
    print("Multi-GPU Training Demo")
    print("=" * 60)

    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("\nCUDA not available, using CPU for demonstration.")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        print(f"\nUsing GPU: {torch.cuda.get_device_name(0)}")

    # Get devices
    devices = list_devices()
    print(f"Available devices: {devices}")

    # For multi-GPU training, you would typically use:
    # torch.distributed.init_process_group(backend="nccl")
    # torch.cuda.set_device(local_rank)

    # Create model
    model = get_model()
    x_train, y_train = get_dataset()

    # Simple training loop
    print("\nRunning simple training loop...")
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Convert data to torch tensors
    x_tensor = torch.from_numpy(x_train).float()
    y_tensor = torch.from_numpy(y_train).long()

    # Training configuration
    batch_size = 32
    epochs = 1

    for epoch in range(epochs):
        # Shuffle data
        indices = torch.randperm(len(x_tensor))
        x_tensor = x_tensor[indices]
        y_tensor = y_tensor[indices]

        total_loss = 0.0
        num_batches = 0

        for i in range(0, len(x_tensor), batch_size):
            x_batch = x_tensor[i : i + batch_size].to(device)
            y_batch = y_tensor[i : i + batch_size].to(device)

            # Forward pass
            # Convert Keras tensor to torch tensor
            logits = model(x_batch, training=True)

            # Compute loss
            loss = loss_fn(y_batch, logits)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.apply(model.trainable_variables)

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"  Epoch {epoch + 1}: Loss = {avg_loss:.4f}")

    print("\nTraining complete!")


def main():
    """Main function to run all demonstrations."""
    print("=" * 60)
    print("Keras PyTorch Distribution with DTensor")
    print("=" * 60)

    # Show backend info
    print(f"\nKeras backend: {keras.config.backend()}")
    print(f"PyTorch version: {torch.__version__}")

    # Run demonstrations
    demonstrate_path_separator_adapter()
    demonstrate_data_parallel()
    demonstrate_model_parallel()
    demonstrate_multi_gpu_training()

    print("\n" + "=" * 60)
    print("All demonstrations complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

