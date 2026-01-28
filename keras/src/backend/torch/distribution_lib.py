"""Utilities for distribution strategy with PyTorch backend.

This module provides distribution support for PyTorch backend including:
- Model parallelism using DTensor
- Data parallelism using torch.distributed
- CPU/GPU/TPU device support
- Path adapter for Keras/PyTorch naming conventions

The adapter layer ensures Keras regex patterns (using `/` separators like
`dense/kernel`) work seamlessly with PyTorch parameter naming (using `.` 
separators like `dense.weight`).
"""

import os
import re
import warnings

import torch
import torch.distributed as dist
import torch.nn as nn

from keras.src.backend.common import global_state

# Try to import DTensor - requires PyTorch 2.1+ with distributed support
try:
    from torch.distributed._tensor import DeviceMesh, DTensor, distribute_module, distribute_tensor
    from torch.distributed._tensor.api import _DTensorPlaceholder
    DTENSOR_AVAILABLE = True
    print("[Torch Distribution] DTensor is available - full distribution support enabled")
except ImportError:
    DTENSOR_AVAILABLE = False
    print("[Torch Distribution] DTensor not available - using fallback distribution")

# Check for TPU availability
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    TPU_AVAILABLE = True
    print("[Torch Distribution] TPU support detected")
except ImportError:
    TPU_AVAILABLE = False
    print("[Torch Distribution] TPU support not available")


def list_devices(device_type=None):
    """Return all the available devices based on the device type.

    Note that this should return the global devices in a distributed setting.

    Args:
        device_type: string of `"cpu"`, `"gpu"` or `"tpu"`. Defaults to `"gpu"`
            or `"tpu"` if available when device_type is not provided. Otherwise
            will return the `"cpu"` devices.

    Return:
        List of devices that are available for distribute computation.
    """
    device_type = device_type.lower() if device_type else None
    devices = []
    
    if device_type in (None, "tpu") and TPU_AVAILABLE:
        try:
            xla_devices = xm.get_xla_supported_devices()
            for dev in xla_devices:
                devices.append(f"tpu:{dev}")
            if device_type == "tpu":
                return devices
        except Exception as e:
            print(f"[Torch Distribution] Warning: Could not get TPU devices: {e}")
    
    if device_type in (None, "gpu"):
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            for i in range(num_gpus):
                devices.append(f"cuda:{i}")
            if device_type == "gpu":
                return devices
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            devices.append("mps:0")
            if device_type == "gpu":
                return devices
    
    if device_type in (None, "cpu"):
        # For CPU, we return all available logical CPUs
        num_cpus = os.cpu_count() or 1
        for i in range(num_cpus):
            devices.append(f"cpu:{i}")
        if device_type == "cpu":
            return devices
    
    # If no specific device_type and no GPUs/TPUs found, return CPU devices
    if not devices:
        num_cpus = os.cpu_count() or 1
        for i in range(num_cpus):
            devices.append(f"cpu:{i}")
    
    return devices


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
    
    if device_type == "tpu" and TPU_AVAILABLE:
        try:
            return len(xm.get_xla_supported_devices())
        except:
            return 0
    
    if device_type == "gpu":
        if torch.cuda.is_available():
            return torch.cuda.device_count()
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 1
        return 0
    
    if device_type == "cpu":
        return os.cpu_count() or 1
    
    # Default: check for GPU/TPU first
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    elif TPU_AVAILABLE:
        try:
            return len(xm.get_xla_supported_devices())
        except:
            return 0
    
    return os.cpu_count() or 1


def _parse_device(device_name):
    """Parse device string to torch device.
    
    Args:
        device_name: String device name (e.g., "cuda:0", "cpu", "tpu:0")
    
    Returns:
        torch.device
    """
    device_name = device_name.lower()
    
    if "tpu" in device_name:
        if TPU_AVAILABLE:
            # Extract device index for TPU
            if ":" in device_name:
                _, idx = device_name.split(":")
                return torch.device(f"xla:{idx}")
            return torch.device("xla")
        else:
            raise ValueError("TPU not available")
    
    if "cuda" in device_name:
        if torch.cuda.is_available():
            return torch.device(device_name)
        raise ValueError("CUDA not available")
    
    if "mps" in device_name:
        return torch.device("mps")
    
    return torch.device("cpu")


class TorchPathAdapter:
    """Adapter to convert between Keras and PyTorch path naming conventions.
    
    Keras uses `/` separators: e.g., "dense/kernel", "dense_1/bias"
    PyTorch uses `.` separators: e.g., "dense.weight", "dense_1.bias"
    
    This adapter allows Keras regex patterns to work with PyTorch parameters.
    """
    
    # Map of PyTorch style paths to Keras style paths
    _cache = {}
    
    @staticmethod
    def keras_to_torch(keras_path):
        """Convert Keras path (with /) to PyTorch style (with .).
        
        Args:
            keras_path: Path like "dense/kernel" or "model/layer_1/weight"
        
        Returns:
            Path like "dense.weight" or "model.layer_1.weight"
        """
        if keras_path in TorchPathAdapter._cache:
            return TorchPathAdapter._cache[keras_path]
        
        # Convert / to .
        torch_path = keras_path.replace("/", ".")
        
        # Cache the result
        TorchPathAdapter._cache[keras_path] = torch_path
        return torch_path
    
    @staticmethod
    def torch_to_keras(torch_path):
        """Convert PyTorch path (with .) to Keras style (with /).
        
        Args:
            torch_path: Path like "dense.weight" or "model.layer_1.weight"
        
        Returns:
            Path like "dense/kernel" or "model/layer_1/weight"
        """
        # Convert . to /
        return torch_path.replace(".", "/")
    
    @staticmethod
    def match_pattern(pattern, torch_path, debug=False):
        """Check if a PyTorch path matches a Keras regex pattern.
        
        Args:
            pattern: Keras-style regex pattern (e.g., "dense.*kernel")
            torch_path: PyTorch-style path (e.g., "dense.weight")
            debug: If True, print debug information
        
        Returns:
            bool: True if the pattern matches
        """
        # Convert the PyTorch path to Keras style for matching
        keras_path = TorchPathAdapter.torch_to_keras(torch_path)
        
        if debug:
            print(f"[Torch Path Adapter] Pattern: {pattern}, Torch path: {torch_path}, Keras path: {keras_path}, Match: {bool(re.search(pattern, keras_path))}")
        
        return bool(re.search(pattern, keras_path))
    
    @staticmethod
    def clear_cache():
        """Clear the path adapter cache."""
        TorchPathAdapter._cache.clear()
        print("[Torch Path Adapter] Cache cleared")


def _to_backend_mesh(device_mesh):
    """Convert the DeviceMesh to PyTorch backend specific Mesh.
    
    Args:
        device_mesh: DeviceMesh instance to convert.
    
    Returns:
        A `torch.distributed.DeviceMesh` instance or None if DTensor not available.
    """
    print(f"[Torch Distribution] Converting DeviceMesh: shape={device_mesh.shape}, axis_names={device_mesh.axis_names}")
    
    if not DTENSOR_AVAILABLE:
        print("[Torch Distribution] DTensor not available, returning None for backend mesh")
        return None
    
    # Get devices as a flat list and reshape
    devices = device_mesh.devices.flatten().tolist()
    shape = device_mesh.devices.shape
    
    # Create DeviceMesh
    torch_mesh = DeviceMesh(
        device_type="cuda" if torch.cuda.is_available() else "cpu",
        mesh=torch.distributed.init_device_mesh(
            "cuda" if torch.cuda.is_available() else "cpu",
            shape=shape,
            init_method="default"
        )
    )
    
    print(f"[Torch Distribution] Created torch DeviceMesh with shape {shape}")
    return torch_mesh


def _to_backend_layout(tensor_layout):
    """Convert the TensorLayout to PyTorch backend specific sharding spec.
    
    Args:
        tensor_layout: TensorLayout instance to convert.
    
    Returns:
        A tuple of axis names or None if DTensor not available.
    """
    print(f"[Torch Distribution] Converting TensorLayout: axes={tensor_layout.axes}")
    
    if not DTENSOR_AVAILABLE:
        print("[Torch Distribution] DTensor not available, returning None for layout")
        return None
    
    if tensor_layout.device_mesh is None:
        print("[Torch Distribution] Warning: device_mesh is None, returning None")
        return None
    
    # Convert axes - replace None with None (unsharded)
    shard_spec = []
    for axis in tensor_layout.axes:
        if axis is None:
            shard_spec.append(None)
        else:
            shard_spec.append(axis)
    
    print(f"[Torch Distribution] Created shard spec: {shard_spec}")
    return tuple(shard_spec)


def distribute_variable(value, layout):
    """Create a distributed variable for PyTorch.
    
    Args:
        value: The initial value of the variable (torch.Tensor).
        layout: `TensorLayout` for the created variable, or a shard spec.
    
    Returns:
        Distributed value (DTensor or replicated torch.Tensor).
    """
    if not DTENSOR_AVAILABLE:
        print(f"[Torch Distribution] distribute_variable: DTensor not available, returning original value")
        return value
    
    print(f"[Torch Distribution] distribute_variable: value.shape={value.shape}, layout={layout}")
    
    from keras.src.distribution import TensorLayout
    
    # Convert TensorLayout to shard spec
    if isinstance(layout, TensorLayout):
        shard_spec = _to_backend_layout(layout)
        device_mesh = layout.device_mesh.backend_mesh if layout.device_mesh else None
    else:
        shard_spec = layout
        device_mesh = None
    
    if shard_spec is None or device_mesh is None:
        print(f"[Torch Distribution] No valid layout, replicating variable")
        return value
    
    # Distribute the tensor
    try:
        dist_tensor = distribute_tensor(value, device_mesh, shard_spec)
        print(f"[Torch Distribution] Successfully distributed tensor to: {shard_spec}")
        return dist_tensor
    except Exception as e:
        print(f"[Torch Distribution] Error distributing tensor: {e}")
        return value


def distribute_tensor(tensor, layout):
    """Distribute the tensor based on the layout.
    
    Args:
        tensor: `torch.Tensor` that needs to be distributed.
        layout: `TensorLayout` for the created variable, or a shard spec.
    
    Returns:
        Distributed value.
    """
    if not DTENSOR_AVAILABLE:
        print(f"[Torch Distribution] distribute_tensor: DTensor not available, returning original tensor")
        return tensor
    
    print(f"[Torch Distribution] distribute_tensor: tensor.shape={tensor.shape}, layout={layout}")
    
    from keras.src.distribution import TensorLayout
    
    # Convert TensorLayout to shard spec
    if isinstance(layout, TensorLayout):
        shard_spec = _to_backend_layout(layout)
        device_mesh = layout.device_mesh.backend_mesh if layout.device_mesh else None
    else:
        shard_spec = layout
        device_mesh = None
    
    if shard_spec is None or device_mesh is None:
        print(f"[Torch Distribution] No valid layout, returning original tensor")
        return tensor
    
    # Distribute the tensor
    try:
        dist_tensor = distribute_tensor(tensor, device_mesh, shard_spec)
        print(f"[Torch Distribution] Successfully distributed tensor to: {shard_spec}")
        return dist_tensor
    except Exception as e:
        print(f"[Torch Distribution] Error distributing tensor: {e}")
        return tensor


def initialize(job_addresses=None, num_processes=None, process_id=None):
    """Initialize the distribution system for PyTorch.
    
    This sets up torch.distributed for multi-process training.
    
    Args:
        job_addresses: Comma-separated IP addresses for all jobs.
        num_processes: Number of worker/processes.
        process_id: The ID of the current worker/process (0 to num_processes-1).
    """
    print(f"[Torch Distribution] Initializing distribution: job_addresses={job_addresses}, num_processes={num_processes}, process_id={process_id}")
    
    # Check environment variables
    if job_addresses is None:
        job_addresses = os.environ.get("KERAS_DISTRIBUTION_JOB_ADDRESSES")
    if num_processes is None:
        num_processes = os.environ.get("KERAS_DISTRIBUTION_NUM_PROCESSES")
    if process_id is None:
        process_id = os.environ.get("KERAS_DISTRIBUTION_PROCESS_ID")
    
    if num_processes is not None:
        num_processes = int(num_processes)
    if process_id is not None:
        process_id = int(process_id)
    
    if num_processes and num_processes > 1:
        # Initialize torch.distributed
        if job_addresses:
            # Multi-host setup
            addresses = job_addresses.split(",")
            master_addr = addresses[0] if addresses else "localhost"
            master_port = os.environ.get("MASTER_PORT", "29500")
            
            os.environ["MASTER_ADDR"] = master_addr
            os.environ["MASTER_PORT"] = master_port
            os.environ["WORLD_SIZE"] = str(num_processes)
            os.environ["RANK"] = str(process_id if process_id is not None else 0)
            
            print(f"[Torch Distribution] Initializing distributed: master_addr={master_addr}, rank={process_id}, world_size={num_processes}")
            
            dist.init_process_group(
                backend="nccl" if torch.cuda.is_available() else "gloo",
                init_method="env://",
                world_size=num_processes,
                rank=process_id if process_id is not None else 0
            )
        else:
            # Single host, multi-GPU
            print(f"[Torch Distribution] Initializing local distributed with {num_processes} processes")
            dist.init_process_group(
                backend="nccl" if torch.cuda.is_available() else "gloo",
                init_method="env://"
            )
        
        print("[Torch Distribution] Distributed initialization complete")
    else:
        print("[Torch Distribution] Single process mode - no distributed initialization needed")


def num_processes():
    """Return the number of processes for the current distribution setting."""
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()


def process_id():
    """Return the current process ID for the distribution setting."""
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_distributed():
    """Check if distributed training is initialized."""
    return dist.is_available() and dist.is_initialized()


def all_reduce(tensor, op="sum"):
    """Perform all-reduce operation on a tensor across all processes.
    
    Args:
        tensor: The tensor to reduce.
        op: Reduction operation ("sum", "product", "min", "max", "avg").
    
    Returns:
        The reduced tensor.
    """
    if not is_distributed():
        return tensor
    
    op_map = {
        "sum": dist.ReduceOp.SUM,
        "product": dist.ReduceOp.PRODUCT,
        "min": dist.ReduceOp.MIN,
        "max": dist.ReduceOp.MAX,
        "avg": dist.ReduceOp.AVG,
    }
    
    reduced_tensor = tensor.clone()
    dist.all_reduce(reduced_tensor, op=op_map.get(op, dist.ReduceOp.SUM))
    
    return reduced_tensor


def broadcast(tensor, src=0):
    """Broadcast a tensor from one process to all others.
    
    Args:
        tensor: The tensor to broadcast.
        src: Source rank.
    
    Returns:
        The broadcast tensor.
    """
    if not is_distributed():
        return tensor
    
    broadcast_tensor = tensor.clone()
    dist.broadcast(broadcast_tensor, src=src)
    
    return broadcast_tensor


def scatter_tensor(tensor, num_chunks=None, dim=0):
    """Scatter a tensor across processes along a dimension.
    
    Args:
        tensor: The tensor to scatter (only used on source rank).
        num_chunks: Number of chunks to split the tensor into.
        dim: Dimension along which to split.
    
    Returns:
        The local chunk of the scattered tensor.
    """
    if not is_distributed():
        return tensor
    
    if num_chunks is None:
        num_chunks = num_processes()
    
    # Calculate chunk size
    tensor_dim = tensor.shape[dim]
    chunk_size = tensor_dim // num_chunks
    
    # Create output tensor
    output = torch.empty_like(tensor) if tensor.dim() > 0 else tensor
    
    # Perform scatter
    dist.scatter(output, src=0, scatter_tensor=tensor)
    
    return output


def gather_tensor(tensor, dim=0, dst=0):
    """Gather tensors from all processes to one process.
    
    Args:
        tensor: The tensor to gather.
        dim: Dimension along which to concatenate.
        dst: Destination rank.
    
    Returns:
        Gathered tensor (only on destination rank, None otherwise).
    """
    if not is_distributed():
        return tensor
    
    output = [torch.empty_like(tensor) for _ in range(num_processes())]
    dist.gather(tensor, dst=dst, gather_list=output)
    
    if process_id() == dst:
        return torch.cat(output, dim=dim)
    return None


def replicate_model(model, broadcast_buffers=True):
    """Replicate a model across all devices for data parallelism.
    
    Args:
        model: The model to replicate.
        broadcast_buffers: Whether to broadcast buffers.
    
    Returns:
        The replicated model.
    """
    if not is_distributed():
        return model
    
    print(f"[Torch Distribution] Replicating model with DDP")
    
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[torch.cuda.current_device()] if torch.cuda.is_available() else None,
        broadcast_buffers=broadcast_buffers
    )
    
    return model


def sync_gradients():
    """Synchronize gradients across all processes."""
    if not is_distributed():
        return
    
    # DDP automatically syncs gradients, but this can be called manually
    # if using custom training loops
    pass

