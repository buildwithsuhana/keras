"""Utilities for distribution strategy with PyTorch backend.

This module provides DTensor-based distribution support for PyTorch backend,
mirroring the JAX backend API for consistency. It supports:
- Data parallelism via DTensor
- Model parallelism via tensor parallelism
- CPU, GPU, and TPU devices
"""

import os

import torch
from torch.distributed.device_mesh import init_device_mesh

try:
    from torch.distributed.tensor import distribute_tensor, DTensor
except ImportError:
    DTensor = None
    distribute_tensor = None


# Try to import torch distributed modules
try:
    import torch.distributed as dist
except ImportError:
    dist = None

try:
    from torch.distributed.tensor import Replicate, Shard
except ImportError:
    Replicate = None
    Shard = None


def _is_distributed_available():
    """Check if torch distributed is available."""
    return dist is not None and dist.is_available()


def list_devices(device_type=None):
    """Return all the available devices based on the device type.

    Note that in a distributed setting, global devices are returned.

    Args:
        device_type: string of `"cpu"`, `"gpu"` or `"tpu"`. Defaults to `"gpu"`
            or `"tpu"` if available when device_type is not provided. Otherwise
            will return the `"cpu"` devices.

    Return:
        List of devices that are available for distribute computation.
    """
    device_type = device_type.lower() if device_type else None

    devices = []

    # Check for TPU
    try:
        import torch_xla.core.xla_model as xm

        if device_type in (None, "tpu"):
            tpu_devices = xm.get_xla_supported_devices("tpu")
            for tpu in tpu_devices:
                devices.append(f"tpu:{tpu}")
    except ImportError:
        pass

    # Check for GPU
    if torch.cuda.is_available():
        if device_type in (None, "gpu"):
            for i in range(torch.cuda.device_count()):
                devices.append(f"cuda:{i}")

    # Check for MPS (Apple Silicon)
    if torch.backends.mps.is_available():
        if device_type in (None, "gpu"):
            devices.append("mps:0")

    # Check for CPU
    if device_type in (None, "cpu") or not devices:
        # Add CPU devices
        num_threads = torch.get_num_threads()
        for i in range(num_threads):
            devices.append(f"cpu:{i}")

    # Filter by device type if specified
    if device_type:
        filtered_devices = [
            d for d in devices if d.startswith(device_type.lower())
        ]
        if filtered_devices:
            return filtered_devices

    return devices if device_type is None else []


def get_device_count(device_type=None):
    """Returns the number of available PyTorch devices.

    Args:
        device_type: Optional device type to count (e.g., "cpu", "gpu", "tpu").
            If `None`, it defaults to counting "gpu" or "tpu" devices if
            available, otherwise it counts "cpu" devices. It does not
            return the sum of all device types.

    Returns:
        int: The total number of PyTorch devices for the specified type.
    """
    device_type = device_type.lower() if device_type else None

    if device_type in (None, "tpu"):
        try:
            import torch_xla.core.xla_model as xm

            tpu_devices = xm.get_xla_supported_devices("tpu")
            if tpu_devices:
                return len(tpu_devices)
        except ImportError:
            pass

    if device_type in (None, "gpu"):
        if torch.cuda.is_available():
            return torch.cuda.device_count()
        if torch.backends.mps.is_available():
            return 1  # MPS doesn't report device count like CUDA

    if device_type == "cpu":
        return torch.get_num_threads()

    # Fallback: count from list_devices
    return len(list_devices(device_type))


def _to_backend_mesh(device_mesh):
    """Convert the Keras DeviceMesh to PyTorch DeviceMesh.

    Args:
        device_mesh: Keras DeviceMesh instance to convert.

    Returns:
        A `torch.distributed.device_mesh.DeviceMesh` instance.
    """
    if not _is_distributed_available():
        raise RuntimeError("torch.distributed is not available")

    # Get device list
    devices = device_mesh.devices.flatten().tolist()

    # Get axis names
    axis_names = device_mesh.axis_names

    # Create the mesh
    # Note: PyTorch DeviceMesh expects a device type as first argument
    torch_mesh = init_device_mesh(
        "cuda" if torch.cuda.is_available() else "cpu",
        device_mesh.shape,
        mesh_dim_names=axis_names,
    )

    return torch_mesh


def _to_backend_layout(tensor_layout):
    """Convert the Keras TensorLayout to PyTorch placements.

    Args:
        tensor_layout: Keras TensorLayout instance to convert.

    Returns:
        A tuple of `Placement` instances for DTensor.
    """
    if Replicate is None or Shard is None:
        raise ImportError("torch.distributed.tensor is not available")

    if tensor_layout.device_mesh is None:
        raise ValueError(
            "Cannot create sharding when device mesh is not set "
            "for TensorLayout."
        )

    mesh = tensor_layout.device_mesh
    placements = []

    for axis in tensor_layout.axes:
        if axis is None:
            placements.append(Replicate())
        else:
            try:
                axis_index = mesh.axis_names.index(axis)
                placements.append(Shard(axis=axis_index))
            except ValueError:
                placements.append(Replicate())

    return tuple(placements)


def distribute_tensor(tensor, layout):
    """Distribute the tensor based on the layout using DTensor.

    Args:
        tensor: torch.Tensor to distribute.
        layout: TensorLayout for the distribution information.

    Returns:
        A `DTensor` instance with the specified layout.
    """
    if distribute_tensor is None:
        raise ImportError("torch.distributed.tensor is not available")

    from keras.src.distribution import TensorLayout

    if isinstance(layout, TensorLayout):
        backend_layout = layout.backend_layout
        placements = _to_backend_layout(layout)
    elif isinstance(layout, tuple):
        # Assume it's already placements
        placements = layout
        backend_layout = None
    else:
        backend_layout = layout
        placements = None

    # Get the mesh
    mesh = None
    if backend_layout is not None and hasattr(backend_layout, "_mesh"):
        mesh = backend_layout._mesh

    if mesh is None and placements is not None:
        # Try to get mesh from placements or create one
        pass

    if placements is not None:
        # Use DTensor directly with placements
        if hasattr(tensor, "_torch_dtensor"):
            return tensor

        # Create DTensor with placements
        device_mesh = _get_default_mesh()
        if device_mesh is not None:
            from torch.distributed.tensor import distribute_tensor as dt

            return dt(tensor, device_mesh, placements)

    # Fallback: return the tensor as is
    return tensor


def distribute_variable(value, layout):
    """Create a distributed variable for PyTorch.

    Args:
        value: The initial value of the variable (torch.Tensor).
        layout: TensorLayout for the created variable.

    Returns:
        A `DTensor` wrapped variable or the original tensor.
    """
    tensor = distribute_tensor(value, layout)

    # If we have a DTensor, we can create a parameter from it
    if DTensor is not None and isinstance(tensor, DTensor):
        # DTensor is already a proper tensor with placement info
        return tensor

    return value


def distribute_data_input(per_process_batch, layout, batch_dim_name):
    """Distribute the input data with the corresponding layout.

    Note that the inputs here is a local worker batch. Within the local worker,
    the data need to be further partitioned to map to each of the devices.

    Args:
        per_process_batch: Tensor that is already sharded to local process size.
        layout: TensorLayout for the distribution information.
        batch_dim_name: Name of the batch dimension axis.

    Returns:
        A global batch distributed according to `layout`.
    """
    from keras.src.distribution import TensorLayout

    if isinstance(layout, TensorLayout):
        layout = layout.backend_layout

    # For DTensor, we can just distribute the tensor
    return distribute_tensor(per_process_batch, layout)


def initialize(job_addresses, num_processes, process_id):
    """Initialize the distribution system for PyTorch backend.

    Args:
        job_addresses: Comma separated IP addresses for all the jobs.
        num_processes: The number of worker/processes.
        process_id: The ID number of the current worker/process.
    """
    if not _is_distributed_available():
        raise RuntimeError("torch.distributed is not available")

    # Set environment variables
    if job_addresses:
        if "," in job_addresses:
            os.environ["MASTER_ADDR"] = job_addresses.split(",")[0]
        else:
            os.environ["MASTER_ADDR"] = job_addresses
    if num_processes:
        os.environ["WORLD_SIZE"] = str(num_processes)
    if process_id is not None:
        os.environ["RANK"] = str(process_id)

    # Set backend
    if torch.cuda.is_available():
        os.environ["BACKEND"] = "nccl"
    else:
        os.environ["BACKEND"] = "gloo"

    # Initialize the process group
    dist.init_process_group(
        backend=os.environ.get("BACKEND", "nccl" if torch.cuda.is_available() else "gloo"),
        init_method="env://",
    )


def num_processes():
    """Return the number of processes for the current distribution setting."""
    if not _is_distributed_available():
        return 1
    return dist.get_world_size()


def process_id():
    """Return the current process ID for the current distribution setting."""
    if not _is_distributed_available():
        return 0
    return dist.get_rank()


def _get_default_mesh():
    """Get the default device mesh from environment."""
    if not _is_distributed_available():
        return None

    # Check if we have a mesh from initialization
    # This is a simplified version; in practice you'd want to cache this
    try:
        world_size = dist.get_world_size()
        if world_size > 1:
            # Create a simple mesh with all available devices
            device_type = "cuda" if torch.cuda.is_available() else "cpu"
            return init_device_mesh(device_type, (world_size,))
    except:
        pass

    return None


# Import helper functions from common backend
from keras.src.backend.common import global_state

GLOBAL_DEVICE_MESH = "torch_device_mesh"


def set_global_device_mesh(mesh):
    """Set the global DeviceMesh for PyTorch backend."""
    global_state.set_global_attribute(GLOBAL_DEVICE_MESH, mesh)


def get_global_device_mesh():
    """Get the global DeviceMesh for PyTorch backend."""
    return global_state.get_global_attribute(GLOBAL_DEVICE_MESH)

