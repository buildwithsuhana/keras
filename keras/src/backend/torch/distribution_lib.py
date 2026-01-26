"""Utilities for distribution strategy with PyTorch backend.

This module provides distribution support for PyTorch backend, similar to what
JAX backend provides. It supports:

1. Data Parallelism: Using torch.nn.DataParallel or DistributedDataParallel
2. Model Parallelism: Manual weight sharding across devices

The implementation integrates with Keras distribution API:
- keras.distribution.DataParallel
- keras.distribution.ModelParallel
- keras.distribution.DeviceMesh
- keras.distribution.TensorLayout
"""

import os
from typing import List, Optional, Union

import torch
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed import DistributedSampler

from keras.src.backend.common import global_state


def list_devices(device_type: Optional[str] = None) -> List[str]:
    """Return all the available devices based on the device type.

    Note: in a distributed setting, global devices are returned.

    Args:
        device_type: string of `"cpu"`, `"gpu"` or `"tpu"`. Defaults to `"gpu"`
            if available when device_type is not provided. Otherwise
            will return the `"cpu"` devices.

    Returns:
        List of devices that are available for distributed computation.
    """
    device_type = device_type.lower() if device_type else None

    if device_type == "tpu":
        # PyTorch doesn't have native TPU support
        return []

    if device_type == "cpu":
        return ["cpu:0"]

    # Default to GPU if available
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        return [f"cuda:{i}" for i in range(num_gpus)]
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        num_xpus = torch.xpu.device_count()
        return [f"xpu:{i}" for i in range(num_xpus)]
    elif torch.backends.mps.is_available():
        return ["mps:0"]
    else:
        return ["cpu:0"]


def get_device_count(device_type: Optional[str] = None) -> int:
    """Returns the number of available PyTorch devices.

    Args:
        device_type: Optional device type to count (e.g., "cpu", "gpu", "tpu").
            If `None`, it defaults to counting "gpu" devices if available,
            otherwise it counts "cpu" devices.

    Returns:
        int: The total number of PyTorch devices for the specified type.
    """
    device_type = device_type.lower() if device_type else None

    if device_type == "tpu":
        return 0

    if device_type == "cpu":
        return 1

    if device_type == "gpu" or device_type is None:
        if torch.cuda.is_available():
            return torch.cuda.device_count()
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.xpu.device_count()
        elif torch.backends.mps.is_available():
            return 1

    return 0


def distribute_variable(value, layout):
    """Create a distributed variable for PyTorch.

    For PyTorch, we handle distribution differently based on the layout:
    - Replicated variables: All devices have a full copy
    - Sharded variables: Parameters are split across devices (for model parallel)

    Args:
        value: The initial value of the variable (torch.Tensor or numpy array).
        layout: `TensorLayout` for the distribution information.

    Returns:
        torch.Tensor with appropriate device placement and distribution.
    """
    from keras.src.distribution import TensorLayout

    if isinstance(layout, TensorLayout):
        # Get the axes from the layout
        axes = layout.axes

        # If all axes are None or only batch axis, replicate
        non_none_axes = [ax for ax in axes if ax is not None]

        if not non_none_axes:
            # Fully replicated - place on first device
            if isinstance(value, torch.Tensor):
                return value.to(f"cuda:0" if torch.cuda.is_available() else "cpu")
            return torch.tensor(value).to(f"cuda:0" if torch.cuda.is_available() else "cpu")

        # For model parallel with sharding, we'll handle this in layer placement
        # The actual sharding is done when layers are moved to devices
        if isinstance(value, torch.Tensor):
            return value.to(f"cuda:0" if torch.cuda.is_available() else "cpu")
        return torch.tensor(value).to(f"cuda:0" if torch.cuda.is_available() else "cpu")

    # If layout is None, just return value with appropriate device
    if isinstance(value, torch.Tensor):
        return value.to(f"cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.tensor(value).to(f"cuda:0" if torch.cuda.is_available() else "cpu")


def distribute_tensor(tensor, layout):
    """Distribute the tensor based on the layout.

    Args:
        tensor: torch.Tensor that needs to be distributed.
        layout: `TensorLayout` for the distribution information.

    Returns:
        Distributed tensor.
    """
    from keras.src.distribution import TensorLayout

    if isinstance(layout, TensorLayout):
        axes = layout.axes
        non_none_axes = [ax for ax in axes if ax is not None]

        if not non_none_axes:
            # Fully replicated
            if isinstance(tensor, torch.Tensor):
                return tensor.to(f"cuda:0" if torch.cuda.is_available() else "cpu")
            return tensor

        # Handle sharding for model parallel
        # For now, place on first device
        if isinstance(tensor, torch.Tensor):
            return tensor.to(f"cuda:0" if torch.cuda.is_available() else "cpu")
        return tensor

    # If no layout, just ensure it's on the right device
    if isinstance(tensor, torch.Tensor):
        return tensor.to(f"cuda:0" if torch.cuda.is_available() else "cpu")
    return tensor


def initialize(
    job_addresses: Optional[str] = None,
    num_processes: Optional[int] = None,
    process_id: Optional[int] = None,
):
    """Initialize the distribution system for multi-process setting.

    Args:
        job_addresses: Comma-separated IP addresses for all the jobs.
        num_processes: Number of processes in the cluster.
        process_id: Current process ID.
    """
    # Get from environment variables if not provided
    if job_addresses is None and "KERAS_DISTRIBUTION_JOB_ADDRESSES" in os.environ:
        job_addresses = os.environ["KERAS_DISTRIBUTION_JOB_ADDRESSES"]
    if num_processes is None and "KERAS_DISTRIBUTION_NUM_PROCESSES" in os.environ:
        num_processes = int(os.environ["KERAS_DISTRIBUTION_NUM_PROCESSES"])
    if process_id is None and "KERAS_DISTRIBUTION_PROCESS_ID" in os.environ:
        process_id = int(os.environ["KERAS_DISTRIBUTION_PROCESS_ID"])

    # Set environment variables for PyTorch
    if job_addresses:
        if "," in job_addresses:
            job_addresses = job_addresses.split(",")
            os.environ["MASTER_ADDR"] = job_addresses[0]
        else:
            os.environ["MASTER_ADDR"] = job_addresses
    else:
        os.environ.setdefault("MASTER_ADDR", "localhost")

    os.environ.setdefault("MASTER_PORT", "29500")

    if num_processes is not None:
        os.environ["WORLD_SIZE"] = str(num_processes)

    if process_id is not None:
        os.environ["RANK"] = str(process_id)

    # Initialize process group
    backend = "nccl" if torch.cuda.is_available() else "gloo"

    init_process_group(
        backend=backend,
        init_method="env://",
        world_size=int(os.environ.get("WORLD_SIZE", 1)),
        rank=int(os.environ.get("RANK", 0)),
    )


def num_processes() -> int:
    """Return the number of processes for the current distribution setting."""
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    return 1


def process_id() -> int:
    """Return the current process ID for the distribution setting."""
    if "RANK" in os.environ:
        return int(os.environ["RANK"])
    return 0


def cleanup():
    """Cleanup the distribution process group."""
    destroy_process_group()


def is_distributed() -> bool:
    """Check if we're in a distributed training setting."""
    return torch.distributed.is_initialized()


def get_local_rank() -> int:
    """Get the local rank for the current process."""
    if is_distributed():
        return int(os.environ.get("LOCAL_RANK", 0))
    return 0


def get_world_size() -> int:
    """Get the world size (number of processes)."""
    if is_distributed():
        return torch.distributed.get_world_size()
    return 1


def get_rank() -> int:
    """Get the global rank of the current process."""
    if is_distributed():
        return torch.distributed.get_rank()
    return 0


def synchronize():
    """Synchronize all processes in distributed training."""
    if is_distributed():
        torch.distributed.barrier()


def _to_backend_device(device_name: str) -> torch.device:
    """Convert a device name to a torch.device.

    Args:
        device_name: Device name like "cuda:0", "cpu", "mps".

    Returns:
        torch.device instance.
    """
    device_name = str(device_name).lower()
    if "gpu" in device_name:
        device_name = device_name.replace("gpu", "cuda")
    return torch.device(device_name)


def _to_backend_mesh(device_mesh):
    """Convert the DeviceMesh to backend-specific representation.

    For PyTorch, we convert to a list of torch devices.

    Args:
        device_mesh: DeviceMesh instance to convert.

    Returns:
        List of torch devices.
    """
    devices = []
    for d in device_mesh.devices.flatten():
        devices.append(_to_backend_device(d))
    return devices


def _to_backend_layout(tensor_layout):
    """Convert the TensorLayout to PyTorch-specific representation.

    For PyTorch, we use a simpler representation since PyTorch's
    DataParallel/DistributedDataParallel handles distribution automatically.

    Args:
        tensor_layout: TensorLayout instance to convert.

    Returns:
        Dict with distribution information.
    """
    if tensor_layout.device_mesh is None:
        raise ValueError(
            "Cannot create layout when device mesh is not set "
            "for TensorLayout."
        )

    # Extract axis names and shape info
    axes = tensor_layout.axes
    mesh_axis_names = tensor_layout.device_mesh.axis_names

    return {
        "axes": axes,
        "mesh_shape": tensor_layout.device_mesh.shape,
        "mesh_axis_names": mesh_axis_names,
    }


# ============================================================================
# Data Parallel Utilities
# ============================================================================

def create_data_parallel_model(
    model,
    device_ids: Optional[List[int]] = None,
    output_device: Optional[int] = None,
    broadcast_buffers: bool = True,
) -> torch.nn.DataParallel:
    """Wrap a model with torch.nn.DataParallel for data parallel training.

    Args:
        model: The Keras model to wrap.
        device_ids: List of device IDs to use. If None, uses all available GPUs.
        output_device: Device to gather outputs. If None, uses first GPU.
        broadcast_buffers: Whether to broadcast buffers at start of training.

    Returns:
        torch.nn.DataParallel wrapped model.
    """
    num_gpus = torch.cuda.device_count()

    if num_gpus <= 1:
        return model

    if device_ids is None:
        device_ids = list(range(num_gpus))

    if output_device is None:
        output_device = device_ids[0]

    return torch.nn.DataParallel(
        model,
        device_ids=device_ids,
        output_device=output_device,
        broadcast_buffers=broadcast_buffers,
    )


def create_distributed_data_parallel_model(
    model,
    device_ids: Optional[List[int]] = None,
    broadcast_buffers: bool = True,
    find_unused_parameters: bool = False,
    bucket_cap_mb: int = 25,
) -> torch.nn.parallel.DistributedDataParallel:
    """Wrap a model with torch.nn.parallel.DistributedDataParallel.

    This is for multi-process training where each process runs on one GPU.

    Args:
        model: The Keras model to wrap.
        device_ids: List of device IDs (should contain one local rank).
        broadcast_buffers: Whether to broadcast buffers at start of training.
        find_unused_parameters: Whether to find unused parameters.
        bucket_cap_mb: Bucket size for gradient reduction.

    Returns:
        torch.nn.parallel.DistributedDataParallel wrapped model.
    """
    local_rank = get_local_rank()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Move model to device
    model = model.to(device)

    if device_ids is None:
        device_ids = [local_rank]

    return torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=device_ids,
        output_device=device if torch.cuda.is_available() else None,
        broadcast_buffers=broadcast_buffers,
        find_unused_parameters=find_unused_parameters,
        bucket_cap_mb=bucket_cap_mb,
    )


def create_distributed_sampler(
    dataset,
    num_replicas: Optional[int] = None,
    rank: Optional[int] = None,
    shuffle: bool = True,
    seed: int = 0,
    drop_last: bool = False,
) -> DistributedSampler:
    """Create a distributed sampler for dataset.

    Args:
        dataset: The dataset to sample from.
        num_replicas: Number of processes participating in distributed training.
        rank: Rank of the current process.
        shuffle: Whether to shuffle the data.
        seed: Random seed for shuffling.
        drop_last: Whether to drop the last incomplete batch.

    Returns:
        torch.utils.data.distributed.DistributedSampler.
    """
    if num_replicas is None:
        num_replicas = get_world_size()
    if rank is None:
        rank = get_rank()

    return DistributedSampler(
        dataset,
        num_replicas=num_replicas,
        rank=rank,
        shuffle=shuffle,
        seed=seed,
        drop_last=drop_last,
    )


def create_dataloader(
    dataset,
    batch_size: int = 32,
    shuffle: bool = False,
    sampler: Optional[DistributedSampler] = None,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> torch.utils.data.DataLoader:
    """Create a DataLoader with optional distributed sampling.

    Args:
        dataset: The dataset to load.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle data (mutually exclusive with sampler).
        sampler: DistributedSampler for distributed training.
        num_workers: Number of worker processes for data loading.
        pin_memory: Whether to pin memory for faster GPU transfer.
        drop_last: Whether to drop last incomplete batch.

    Returns:
        torch.utils.data.DataLoader.
    """
    if sampler is not None:
        shuffle = False  # Shuffle is handled by sampler

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=drop_last,
    )


# ============================================================================
# Model Parallel Utilities
# ============================================================================

def get_device_for_layer(layer_index: int, num_layers: int, num_devices: int) -> int:
    """Get the device to place a layer on for model parallelism.

    Args:
        layer_index: Index of the layer.
        num_layers: Total number of layers to distribute.
        num_devices: Number of available devices.

    Returns:
        Device index to place the layer on.
    """
    if num_devices <= 1:
        return 0
    layers_per_device = max(1, num_layers // num_devices)
    return min(layer_index // layers_per_device, num_devices - 1)


def shard_model_by_layers(
    model,
    layer_devices: dict,
) -> None:
    """Manually shard model layers across devices.

    This is for model parallelism where different layers are placed
    on different devices.

    Args:
        model: The Keras model to shard.
        layer_devices: Dict mapping layer names to device indices.
    """
    for layer in model.layers:
        layer_path = layer.name
        if layer_path in layer_devices:
            device = layer_devices[layer_path]
            if torch.cuda.is_available():
                device = f"cuda:{device}"
            else:
                device = "cpu"
            # Move layer to specified device
            layer.to(device)


# ============================================================================
# Utility Functions
# ============================================================================

def get_default_device() -> torch.device:
    """Get the default device for the current process."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_current_device() -> torch.device:
    """Get the current device for the current process in distributed setting."""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{get_local_rank()}")
    return get_default_device()


def move_model_to_device(model, device: torch.device) -> torch.nn.Module:
    """Move a model to the specified device.

    Args:
        model: The model to move.
        device: Target device.

    Returns:
        Model on the target device.
    """
    if isinstance(model, torch.nn.Module):
        return model.to(device)
    return model


def get_layer_device_placement(layer) -> str:
    """Get the device placement for a layer.

    Args:
        layer: Keras layer instance.

    Returns:
        Device string like "cuda:0" or "cpu".
    """
    # Check if layer has any variables
    if hasattr(layer, "variables") and len(layer.variables) > 0:
        for var in layer.variables:
            if hasattr(var, "value") and isinstance(var.value, torch.Tensor):
                return str(var.value.device)
    return str(get_default_device())


def get_gradients(model) -> dict:
    """Get gradients from all trainable parameters.

    Args:
        model: The model to get gradients from.

    Returns:
        Dict mapping parameter names to gradients.
    """
    gradients = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            gradients[name] = param.grad
    return gradients


def average_gradients(gradients: dict) -> dict:
    """Average gradients across all processes in distributed training.

    Args:
        gradients: Dict mapping parameter names to gradients.

    Returns:
        Dict with averaged gradients.
    """
    if not is_distributed():
        return gradients

    for name in gradients:
        grad = gradients[name]
        # Average across all processes
        torch.distributed.all_reduce(grad, op=torch.distributed.ReduceOp.SUM)
        grad.div_(get_world_size())

    return gradients

