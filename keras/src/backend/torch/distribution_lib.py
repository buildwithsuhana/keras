"""Utilities for distribution strategy with PyTorch backend.

This module provides DTensor-based distribution support for PyTorch backend,
mirroring the JAX backend API for consistency. It supports:
- Data parallelism via DTensor
- Model parallelism via tensor parallelism
- CPU, GPU, and TPU devices
"""

import logging
import os

import torch
from torch.distributed.device_mesh import init_device_mesh

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a handler if none exists
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

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
    
    logger.info(f"Looking for devices of type: {device_type}")

    # Check for TPU
    try:
        import torch_xla.core.xla_model as xm

        if device_type in (None, "tpu"):
            tpu_devices = xm.get_xla_supported_devices("tpu")
            logger.info(f"Found {len(tpu_devices)} TPU devices")
            for tpu in tpu_devices:
                devices.append(f"tpu:{tpu}")
    except ImportError:
        logger.debug("torch_xla not available, skipping TPU detection")

    # Check for GPU
    if torch.cuda.is_available():
        if device_type in (None, "gpu"):
            for i in range(torch.cuda.device_count()):
                devices.append(f"cuda:{i}")
            logger.info(f"Found {torch.cuda.device_count()} GPU devices")
    else:
        logger.info("No CUDA GPUs available")

    # Check for MPS (Apple Silicon)
    if torch.backends.mps.is_available():
        if device_type in (None, "gpu"):
            devices.append("mps:0")
            logger.info("Found MPS device (Apple Silicon)")

    # Check for CPU
    if device_type in (None, "cpu") or not devices:
        # Add CPU devices
        num_threads = torch.get_num_threads()
        logger.info(f"Found {num_threads} CPU threads")
        for i in range(num_threads):
            devices.append(f"cpu:{i}")

    # Filter by device type if specified
    if device_type:
        filtered_devices = [
            d for d in devices if d.startswith(device_type.lower())
        ]
        if filtered_devices:
            logger.info(f"Filtered to {len(filtered_devices)} devices of type {device_type}")
            return filtered_devices
        else:
            logger.warning(f"No devices found of type {device_type}")

    logger.info(f"Total devices found: {len(devices)}")
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
    
    logger.info(f"Getting device count for type: {device_type}")

    if device_type in (None, "tpu"):
        try:
            import torch_xla.core.xla_model as xm

            tpu_devices = xm.get_xla_supported_devices("tpu")
            if tpu_devices:
                logger.info(f"Found {len(tpu_devices)} TPU devices")
                return len(tpu_devices)
        except ImportError:
            logger.debug("torch_xla not available")

    if device_type in (None, "gpu"):
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            logger.info(f"Found {count} GPU devices")
            return count
        if torch.backends.mps.is_available():
            logger.info("Found MPS device (counting as 1)")
            return 1  # MPS doesn't report device count like CUDA

    if device_type == "cpu":
        count = torch.get_num_threads()
        logger.info(f"Found {count} CPU threads")
        return count

    # Fallback: count from list_devices
    count = len(list_devices(device_type))
    logger.info(f"Found {count} devices of type {device_type}")
    return count


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

    logger.info(f"Converting Keras DeviceMesh to PyTorch DeviceMesh")
    logger.info(f"  Shape: {device_mesh.shape}")
    logger.info(f"  Axis names: {axis_names}")
    logger.info(f"  Devices: {devices}")

    # Create the mesh
    # Note: PyTorch DeviceMesh expects a device type as first argument
    torch_mesh = init_device_mesh(
        "cuda" if torch.cuda.is_available() else "cpu",
        device_mesh.shape,
        mesh_dim_names=axis_names,
    )
    
    logger.info(f"Successfully created PyTorch DeviceMesh")
    logger.info(f"  Mesh shape: {torch_mesh.shape}")
    logger.info(f"  Mesh dimensions: {list(torch_mesh.shape.keys())}")

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

    logger.info(f"Converting TensorLayout to PyTorch placements")
    logger.info(f"  Axes: {tensor_layout.axes}")
    logger.info(f"  Device mesh shape: {mesh.shape}")
    logger.info(f"  Device mesh axis names: {mesh.axis_names}")

    for axis in tensor_layout.axes:
        if axis is None:
            placements.append(Replicate())
            logger.debug(f"  Axis {axis}: Replicate()")
        else:
            try:
                axis_index = mesh.axis_names.index(axis)
                placements.append(Shard(axis=axis_index))
                logger.debug(f"  Axis {axis}: Shard(axis={axis_index})")
            except ValueError:
                placements.append(Replicate())
                logger.debug(f"  Axis {axis}: Replicate() (fallback)")

    logger.info(f"Converted to placements: {placements}")
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

    logger.info(f"Distributing tensor with shape: {tensor.shape}")
    logger.info(f"  Layout: {layout}")
    
    if isinstance(layout, TensorLayout):
        backend_layout = layout.backend_layout
        placements = _to_backend_layout(layout)
        logger.info(f"  Backend layout: {backend_layout}")
        logger.info(f"  Placements: {placements}")
    elif isinstance(layout, tuple):
        # Assume it's already placements
        placements = layout
        backend_layout = None
        logger.info(f"  Using provided placements: {placements}")
    else:
        backend_layout = layout
        placements = None
        logger.info(f"  Using backend layout: {backend_layout}")

    # Get the mesh
    mesh = None
    if backend_layout is not None and hasattr(backend_layout, "_mesh"):
        mesh = backend_layout._mesh
        logger.info(f"  Mesh from backend layout: {mesh}")

    if mesh is None and placements is not None:
        # Try to get mesh from placements or create one
        pass

    if placements is not None:
        # Use DTensor directly with placements
        if hasattr(tensor, "_torch_dtensor"):
            logger.debug("  Tensor is already a DTensor")
            return tensor

        # Create DTensor with placements
        device_mesh = _get_default_mesh()
        logger.info(f"  Device mesh: {device_mesh}")
        
        if device_mesh is not None:
            from torch.distributed.tensor import distribute_tensor as dt

            logger.info(f"  Creating DTensor with placements: {placements}")
            result = dt(tensor, device_mesh, placements)
            logger.info(f"  Result type: {type(result)}")
            logger.info(f"  Result shape: {result.shape if hasattr(result, 'shape') else 'N/A'}")
            return result

    # Fallback: return the tensor as is
    logger.info(f"  Returning tensor as-is (no distribution applied)")
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

    logger.info("Initializing PyTorch distributed training")
    logger.info(f"  Job addresses: {job_addresses}")
    logger.info(f"  Number of processes: {num_processes}")
    logger.info(f"  Process ID: {process_id}")

    # Set environment variables
    if job_addresses:
        if "," in job_addresses:
            os.environ["MASTER_ADDR"] = job_addresses.split(",")[0]
        else:
            os.environ["MASTER_ADDR"] = job_addresses
        logger.info(f"  Master address: {os.environ['MASTER_ADDR']}")
    if num_processes:
        os.environ["WORLD_SIZE"] = str(num_processes)
        logger.info(f"  World size: {os.environ['WORLD_SIZE']}")
    if process_id is not None:
        os.environ["RANK"] = str(process_id)
        logger.info(f"  Rank: {os.environ['RANK']}")

    # Set backend
    if torch.cuda.is_available():
        os.environ["BACKEND"] = "nccl"
        logger.info("  Using NCCL backend (GPU)")
    else:
        os.environ["BACKEND"] = "gloo"
        logger.info("  Using Gloo backend (CPU)")

    # Initialize the process group
    logger.info("Initializing process group...")
    dist.init_process_group(
        backend=os.environ.get("BACKEND", "nccl" if torch.cuda.is_available() else "gloo"),
        init_method="env://",
    )
    logger.info("Process group initialized successfully")
    
    # Log additional info
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    logger.info(f"  World size: {world_size}")
    logger.info(f"  Current rank: {rank}")


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


def log_variable_layout(variable_path, layout, matched_key=None):
    """Log information about variable layout assignment.
    
    Args:
        variable_path: The path to the variable (e.g., "dense/kernel")
        layout: The TensorLayout that was assigned
        matched_key: The key in LayoutMap that matched (if any)
    """
    logger.info(f"Variable layout assignment:")
    logger.info(f"  Variable path: {variable_path}")
    if matched_key:
        logger.info(f"  Matched pattern: {matched_key}")
    logger.info(f"  Layout axes: {layout.axes if layout else 'None'}")
    if layout and layout.device_mesh:
        logger.info(f"  Device mesh shape: {layout.device_mesh.shape}")
        logger.info(f"  Device mesh axes: {layout.device_mesh.axis_names}")

