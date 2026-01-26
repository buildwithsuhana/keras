"""Simple Multi-GPU Training with PyTorch Backend.

This example demonstrates how to use data parallelism and model parallelism
with Keras when using the PyTorch backend, without needing complex DTensor setup.

## Data Parallel

For data parallel, PyTorch's `torch.nn.DataParallel` or 
`torch.nn.parallel.DistributedDataParallel` automatically handle:
- Replicating the model on each GPU
- Splitting the input batch across GPUs
- Merging outputs from all GPUs
- Synchronizing gradients

## Model Parallel

For model parallel, we manually split the model weights across devices.
This is useful when the model is too large to fit on a single GPU.
"""

import os

# Set the backend to PyTorch
os.environ["KERAS_BACKEND"] = "torch"

import torch
import numpy as np
import keras

# Configure for multi-GPU if available
device_type = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device type: {device_type}")

num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
print(f"Number of GPUs: {num_gpus}")


def get_model():
    """Create a simple convnet model."""
    inputs = keras.Input(shape=(28, 28, 1))
    x = keras.layers.Rescaling(1.0 / 255.0)(inputs)
    x = keras.layers.Conv2D(32, 3, activation="relu")(x)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Conv2D(64, 3, activation="relu")(x)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    outputs = keras.layers.Dense(10)(x)
    model = keras.Model(inputs, outputs)
    return model


def get_dataset():
    """Load and prepare MNIST dataset."""
    (x_train, y_train), _ = keras.datasets.mnist.load_data()
    x_train = np.expand_dims(x_train.astype("float32"), -1)
    x_train = x_train / 255.0
    # Use a small subset for quick demonstration
    x_train = x_train[:1000]
    y_train = y_train[:1000]
    return x_train, y_train


# =============================================================================
# DATA PARALLEL EXAMPLE
# =============================================================================

def data_parallel_training():
    """Train using PyTorch DataParallel."""
    print("\n=== Data Parallel Training ===")
    
    # Load data
    x_train, y_train = get_dataset()
    
    # Create model
    model = get_model()
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    
    # Wrap model with PyTorch DataParallel if multiple GPUs
    if num_gpus > 1:
        print(f"Wrapping model with DataParallel on {num_gpus} GPUs...")
        model = torch.nn.DataParallel(model)
        model.to(device_type)
    
    # Train
    print("Training...")
    model.fit(
        x_train, y_train,
        epochs=2,
        batch_size=32,
        validation_split=0.1,
        verbose=1,
    )
    
    print("Data parallel training complete!")
    return model


# =============================================================================
# DISTRIBUTED DATA PARALLEL (MULTI-PROCESS)
# =============================================================================

def distributed_data_parallel_training():
    """Train using PyTorch DistributedDataParallel for multi-process training.
    
    This is more robust than DataParallel and works across multiple machines.
    """
    print("\n=== Distributed Data Parallel Training ===")
    
    import torch.multiprocessing as mp
    from torch.utils.data.distributed import DistributedSampler
    from torch.nn.parallel import DistributedDataParallel as DDP
    
    def setup_process(rank, world_size):
        """Setup distributed process group."""
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        torch.distributed.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            world_size=world_size,
            rank=rank,
        )
    
    def cleanup_process():
        """Cleanup distributed process group."""
        torch.distributed.destroy_process_group()
    
    def train_fn(rank, world_size):
        """Training function for each process."""
        setup_process(rank, world_size)
        
        # Load data
        x_train, y_train = get_dataset()
        
        # Create model
        model = get_model()
        model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )
        
        # Wrap with DDP
        device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model = DDP(model, device_ids=[rank] if torch.cuda.is_available() else None)
        
        # Train
        print(f"Process {rank}: Training...")
        model.fit(
            x_train, y_train,
            epochs=1,
            batch_size=16,
            verbose=0,
        )
        print(f"Process {rank}: Training complete!")
        
        cleanup_process()
    
    if num_gpus > 1:
        print(f"Starting {num_gpus} processes for distributed training...")
        mp.spawn(
            train_fn,
            args=(num_gpus,),
            nprocs=num_gpus,
            join=True,
        )
    else:
        print("Only 1 GPU available, skipping distributed training.")
    
    print("Distributed data parallel training complete!")


# =============================================================================
# MODEL PARALLEL EXAMPLE (Manual Weight Sharding)
# =============================================================================

def model_parallel_training():
    """Train using manual model parallelism.
    
    This splits the model across multiple GPUs when the model is too large
    to fit on a single GPU.
    """
    print("\n=== Model Parallel Training ===")
    
    if num_gpus < 2:
        print("Model parallel requires at least 2 GPUs. Skipping...")
        return
    
    # For model parallel, we manually place different parts of the model
    # on different devices
    
    class ModelParallelModel(keras.Model):
        """A model that places different layers on different devices."""
        
        def __init__(self):
            super().__init__()
            # First part of the model on GPU 0
            self.conv1 = keras.layers.Conv2D(32, 3, activation="relu")
            self.pool1 = keras.layers.MaxPooling2D()
            self.conv2 = keras.layers.Conv2D(64, 3, activation="relu")
            self.pool2 = keras.layers.MaxPooling2D()
            
            # Second part of the model on GPU 1
            self.flatten = keras.layers.Flatten()
            self.dense1 = keras.layers.Dense(128, activation="relu")
            self.dense2 = keras.layers.Dense(10)
            
        def call(self, inputs, training=False):
            # Move input to GPU 0
            inputs = inputs.to(f"cuda:0")
            
            # Process on GPU 0
            x = self.conv1(inputs)
            x = self.pool1(x)
            x = self.conv2(x)
            x = self.pool2(x)
            
            # Move to GPU 1 for remaining layers
            x = x.to(f"cuda:1")
            
            x = self.flatten(x)
            x = self.dense1(x)
            x = self.dense2(x)
            return x
    
    # Create and train model
    model = ModelParallelModel()
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    
    x_train, y_train = get_dataset()
    
    print("Training with manual model parallelism...")
    model.fit(
        x_train, y_train,
        epochs=2,
        batch_size=16,
        verbose=1,
    )
    
    print("Model parallel training complete!")


# =============================================================================
# USING KERAS DISTRIBUTION API (SIMPLE WRAPPER)
# =============================================================================

def keras_distribution_api():
    """Use Keras distribution API with PyTorch backend.
    
    Note: For PyTorch backend, the distribution API provides a simple
    wrapper that handles device placement and data sharding.
    """
    print("\n=== Keras Distribution API ===")
    
    from keras.distribution import (
        list_devices, 
        DataParallel, 
        DeviceMesh, 
        TensorLayout,
    )
    
    # List available devices
    devices = list_devices("gpu")
    print(f"Available GPU devices: {devices}")
    
    # Create a simple data parallel distribution
    if len(devices) > 1:
        # Create device mesh with all GPUs
        device_mesh = DeviceMesh(
            shape=(len(devices),),
            axis_names=["batch"],
            devices=devices,
        )
        print(f"Created device mesh: {device_mesh}")
        
        # Use data parallel distribution
        distribution = DataParallel(device_mesh=device_mesh)
        print(f"Using distribution: {distribution}")
    
    # Train model
    model = get_model()
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    
    x_train, y_train = get_dataset()
    
    print("Training with Keras distribution API...")
    model.fit(
        x_train, y_train,
        epochs=1,
        batch_size=32,
        verbose=1,
    )
    
    print("Keras distribution API training complete!")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Run all examples
    data_parallel_training()
    
    # Uncomment to run distributed training (requires multiple GPUs)
    # distributed_data_parallel_training()
    
    if num_gpus >= 2:
        model_parallel_training()
    
    keras_distribution_api()
    
    print("\n=== All Examples Complete ===")

