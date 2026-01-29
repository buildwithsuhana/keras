"""Utilities for distribution strategy with PyTorch backend using DTensor.

This module provides distribution support for PyTorch backend using PyTorch's
DTensor API. DTensor is a PyTorch API for distributed tensor computing that
provides a unified interface for both data parallelism and model parallelism.

Key features:
- Device mesh creation and management
- Tensor layout specification and distribution
- Support for CPU, GPU, and TPU devices
- Path adapter for converting between Keras `/` paths and PyTorch `.` paths
- Debug logging for troubleshooting

Example usage:
    import torch
    from keras.src.distribution import DeviceMesh, TensorLayout, LayoutMap, ModelParallel
    
    # Create a device mesh for 8 GPUs (4 data parallel, 2 model parallel)
    devices = [f"cuda:{i}" for i in range(8)]
    device_mesh = DeviceMesh(shape=(4, 2), axis_names=["data", "model"], devices=devices)
    
    # Create a layout map for model parallelism
    layout_map = LayoutMap(device_mesh)
    layout_map['dense.*kernel'] = (None, 'model')
    layout_map['dense.*bias'] = ('model',)
    
    # Set up the distribution
    distribution = ModelParallel(layout_map=layout_map, batch_dim_name="data")
    with distribution.scope():
        model = create_model()
        model.compile(...)
        model.fit(...)
"""

import logging
import os
import re
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from keras.src.backend.common import global_state

# Set up debug logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Try to import DTensor - it may not be available in all PyTorch versions
try:
    from torch.distributed._tensor import (
        DTensor,
        DeviceMesh,
        Placement,
        Replicate,
        Shard,
    )
    from torch.distributed._tensor.api import distribute_tensor as torch_distribute_tensor
    from torch.distributed._tensor import DeviceMesh as TorchDeviceMesh
    DTENSOR_AVAILABLE = True
except ImportError:
    DTENSOR_AVAILABLE = False
    torch_distribute_tensor = None
    logger.warning(
        "PyTorch DTensor is not available. Distribution support will be limited. "
        "Please install PyTorch with DTensor support (torch>=2.1.0)."
    )
from keras.src.backend.torch.core import convert_to_tensor

# Global variable to track if distribution is initialized
_DISTRIBUTION_INITIALIZED = False


def _get_debug_setting() -> bool:
    """Get debug setting from environment variable."""
    return os.environ.get("KERAS_DISTRIBUTION_DEBUG", "0") == "1"


class TorchPlacement:
    """Represents a placement type for DTensor.
    
    This class wraps PyTorch's placement types (Replicate, Shard) to provide
    a consistent interface with JAX/TensorFlow backends.
    """
    
    def __init__(self, placement_type: str, **kwargs):
        """Initialize a placement.
        
        Args:
            placement_type: Type of placement - 'replicate' or 'shard'
            **kwargs: Additional arguments for shard placement
                - shard_dim: Dimension to shard on (for 'shard' placement)
        """
        self._placement_type = placement_type
        self._shard_dim = kwargs.get('shard_dim', 0)
        
        if placement_type == 'replicate':
            self._placement = Replicate()
        elif placement_type == 'shard':
            self._placement = Shard(dim=self._shard_dim)
        else:
            raise ValueError(f"Unknown placement type: {placement_type}")
    
    @property
    def placement_type(self) -> str:
        return self._placement_type
    
    @property
    def shard_dim(self) -> int:
        return self._shard_dim
    
    @property
    def torch_placement(self):
        return self._placement
    
    def __repr__(self):
        if self._placement_type == 'replicate':
            return "Replicate()"
        else:
            return f"Shard(dim={self._shard_dim})"


def _parse_device(device_name: str) -> torch.device:
    """Parse a device name string to a torch.device.
    
    Args:
        device_name: Device name string like 'cpu:0', 'cuda:0', 'cuda', 'mps:0'
    
    Returns:
        torch.device object
    """
    device_name = str(device_name).lower()
    
    # Handle special cases
    if device_name == 'cpu':
        return torch.device('cpu')
    elif device_name == 'cuda' or device_name.startswith('cuda:'):
        if torch.cuda.is_available():
            return torch.device(device_name if ':' in device_name else 'cuda:0')
        else:
            raise RuntimeError("CUDA is not available")
    elif device_name.startswith('mps'):
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            raise RuntimeError("MPS is not available")
    elif device_name.startswith('xpu'):
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            return torch.device(device_name if ':' in device_name else 'xpu:0')
        else:
            raise RuntimeError("XPU is not available")
    else:
        return torch.device(device_name)


def list_devices(device_type: Optional[str] = None) -> List[str]:
    """Return all the available devices based on the device type.

    Note that this should return the global devices in a distributed setting.

    Args:
        device_type: string of "cpu", "gpu", "tpu", or None for all.
            When None, returns GPU/TPU if available, otherwise CPU.
            Note: "gpu" is an alias for "cuda" in PyTorch.

    Returns:
        List of device strings like ['cuda:0', 'cuda:1', ...]
    """
    if _get_debug_setting():
        logger.debug(f"list_devices called with device_type={device_type}")
    
    device_type = device_type.lower() if device_type else None
    
    devices = []
    
    # Check for TPU (via libtpu)
    if device_type in (None, 'tpu'):
        try:
            import torch_xla.core.xla_model as xm
            tpu_devices = xm.get_xla_supported_devices('tpu')
            devices.extend([f'tpu:{i}' for i in range(len(tpu_devices))])
            if device_type == 'tpu':
                return devices
        except ImportError:
            pass  # TPU not available
    
    # Check for GPU/CUDA
    if device_type in (None, 'gpu', 'cuda'):
        if torch.cuda.is_available():
            nvidia_devices = torch.cuda.device_count()
            devices.extend([f'cuda:{i}' for i in range(nvidia_devices)])
            if device_type in ('gpu', 'cuda'):
                return devices
        # Check for MPS (Apple Silicon GPU)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            devices.append('mps:0')
            if device_type == 'gpu':
                return devices
        # Check for XPU (Intel GPU)
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            xpu_devices = torch.xpu.device_count()
            devices.extend([f'xpu:{i}' for i in range(xpu_devices)])
            if device_type == 'gpu':
                return devices
    
    # Check for CPU
    if device_type in (None, 'cpu'):
        # For CPU, we simulate multiple devices for parallel execution
        num_cpu = os.cpu_count() or 4
        devices.extend([f'cpu:{i}' for i in range(num_cpu)])
        if device_type == 'cpu':
            return devices
    
    # If no specific device type was requested and we found GPU/TPU, return those
    if device_type is None:
        # Remove CPU devices if we found other devices
        non_cpu_devices = [d for d in devices if not d.startswith('cpu:')]
        if non_cpu_devices:
            return non_cpu_devices
    
    return devices


def get_device_count(device_type: Optional[str] = None) -> int:
    """Returns the number of available devices.
    
    Args:
        device_type: Optional device type to count (e.g., "cpu", "gpu", "tpu").
            If None, counts GPU/TPU if available, otherwise CPU.
    
    Returns:
        int: The total number of devices of the specified type.
    """
    if _get_debug_setting():
        logger.debug(f"get_device_count called with device_type={device_type}")
    
    device_type = device_type.lower() if device_type else None
    
    # TPU
    if device_type in (None, 'tpu'):
        try:
            import torch_xla.core.xla_model as xm
            tpu_devices = xm.get_xla_supported_devices('tpu')
            count = len(tpu_devices)
            if device_type == 'tpu':
                return count
            if count > 0:
                return count
        except ImportError:
            pass
    
    # GPU/CUDA
    if device_type in (None, 'gpu', 'cuda'):
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            if device_type in ('gpu', 'cuda'):
                return count
            if count > 0:
                return count
        # MPS
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            if device_type in ('gpu', 'cuda'):
                return 1
            if device_type is None:
                return 1
        # XPU
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            count = torch.xpu.device_count()
            if device_type in ('gpu', 'cuda'):
                return count
            if count > 0:
                return count
    
    # CPU
    if device_type in (None, 'cpu'):
        return os.cpu_count() or 1
    
    return 0


def initialize(
    job_addresses: Optional[str] = None,
    num_processes: Optional[int] = None,
    process_id: Optional[int] = None,
) -> None:
    """Initialize the distribution system for multi-process settings.
    
    This function initializes PyTorch's distributed communication for
    multi-process/multi-device training.
    
    Args:
        job_addresses: Comma-separated IP addresses for all jobs in the cluster.
            For single machine with multiple GPUs, this can be None.
        num_processes: Number of worker processes in the cluster.
        process_id: The ID of the current worker (0 to num_processes-1).
    """
    global _DISTRIBUTION_INITIALIZED
    
    if _get_debug_setting():
        logger.debug(
            f"initialize called with job_addresses={job_addresses}, "
            f"num_processes={num_processes}, process_id={process_id}"
        )
    
    # FIRST: Check if running with torchrun (common in multi-GPU training)
    # torchrun sets LOCAL_RANK and WORLD_SIZE environment variables BEFORE
    # the script starts, and also initializes torch.distributed
    local_rank = os.environ.get("LOCAL_RANK")
    world_size_env = os.environ.get("WORLD_SIZE")
    
    if local_rank is not None and world_size_env is not None:
        # Running with torchrun or similar distributed launcher
        if num_processes is None:
            num_processes = int(world_size_env)
        if process_id is None:
            process_id = int(local_rank)
        if _get_debug_setting():
            logger.debug(
                f"Detected torchrun environment: LOCAL_RANK={local_rank}, "
                f"WORLD_SIZE={world_size_env}"
            )
    
    # Check for environment variables (Keras-specific)
    if job_addresses is None:
        job_addresses = os.environ.get("KERAS_DISTRIBUTION_JOB_ADDRESSES")
    if num_processes is None:
        num_processes_str = os.environ.get("KERAS_DISTRIBUTION_NUM_PROCESSES")
        if num_processes_str:
            num_processes = int(num_processes_str)
    if process_id is None:
        process_id_str = os.environ.get("KERAS_DISTRIBUTION_PROCESS_ID")
        if process_id_str:
            process_id = int(process_id_str)
    
    # For single-process multi-device, no special initialization needed
    # But first check if torchrun has already initialized torch.distributed
    if num_processes is None or num_processes == 1:
        # Check if torchrun already initialized distributed
        if torch.distributed.is_initialized():
            _DISTRIBUTION_INITIALIZED = True
            if _get_debug_setting():
                logger.debug(
                    f"torch.distributed already initialized by torchrun: "
                    f"rank={torch.distributed.get_rank()}, "
                    f"world_size={torch.distributed.get_world_size()}"
                )
        else:
            _DISTRIBUTION_INITIALIZED = True
            if _get_debug_setting():
                logger.debug("Single-process mode - no distributed initialization needed")
        return
    
    # For multi-process, initialize PyTorch distributed
    # First check if torchrun has already initialized it
    if torch.distributed.is_initialized():
        _DISTRIBUTION_INITIALIZED = True
        if _get_debug_setting():
            logger.debug(
                f"Torch distributed already initialized: "
                f"rank={torch.distributed.get_rank()}, "
                f"world_size={torch.distributed.get_world_size()}"
            )
    else:
        if job_addresses and "," in job_addresses:
            # Multiple addresses provided
            addresses = job_addresses.split(",")
            init_method = f"tcp://{addresses[0]}"
        elif job_addresses:
            # Single address provided
            init_method = f"tcp://{job_addresses}"
        else:
            # Use environment
            init_method = "env://"
        
        world_size = num_processes if num_processes else -1
        rank = process_id if process_id is not None else 0
        
        torch.distributed.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            init_method=init_method,
            world_size=world_size,
            rank=rank,
        )
        
        if _get_debug_setting():
            logger.debug(
                f"Distributed process group initialized: "
                f"rank={rank}, world_size={world_size}"
            )
    
    _DISTRIBUTION_INITIALIZED = True


def num_processes() -> int:
    """Return the number of processes for the current distribution setting."""
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def process_id() -> int:
    """Return the current process ID for the distribution setting."""
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def distribute_tensor(tensor: torch.Tensor, layout) -> torch.Tensor:
    """Distribute the tensor based on the layout.
    
    Args:
        tensor: torch.Tensor to distribute.
        layout: TensorLayout or DTensor placement specification.
            Can be:
            - TensorLayout: Keras layout object
            - tuple: (axis_names,) specifying sharding
            - None: replicate the tensor
    
    Returns:
        Distributed torch.Tensor or DTensor
    """
    if _get_debug_setting():
        logger.debug(f"distribute_tensor called with tensor shape={tensor.shape}, layout={layout}")
    
    if layout is None:
        # Replicate the tensor
        if DTENSOR_AVAILABLE and _DISTRIBUTION_INITIALIZED:
            # Use DTensor for replication
            device_mesh = _get_default_device_mesh()
            if device_mesh is not None:
                return DTensor.from_local(tensor, device_mesh, [Replicate()])
        # Return tensor with requires_grad preserved for gradient computation
        if tensor.requires_grad:
            return tensor
        # Enable gradients for floating point tensors (for training)
        if tensor.dtype.is_floating_point:
            return tensor.requires_grad_(True)
        return tensor
    
    # Handle TensorLayout
    if hasattr(layout, 'backend_layout'):
        # It's a Keras TensorLayout
        backend_layout = layout.backend_layout
    else:
        backend_layout = layout
    
    # Handle DTensor sharding
    if DTENSOR_AVAILABLE:
        device_mesh = _get_default_device_mesh()
        
        if device_mesh is not None:
            if hasattr(backend_layout, 'placements'):
                # It's a DTensor Layout
                if isinstance(tensor, DTensor):
                    # Already a DTensor, just redistribute
                    return redistribute_dtensor(tensor, device_mesh, backend_layout)
                else:
                    # Convert to DTensor
                    return torch_distribute_tensor(tensor, device_mesh, backend_layout.placements)
            elif isinstance(backend_layout, tuple):
                # Tuple of axis names
                placements = _axis_names_to_placements(backend_layout, device_mesh)
                if isinstance(tensor, DTensor):
                    return redistribute_dtensor(tensor, device_mesh, placements)
                return distribute_tensor(tensor, device_mesh, placements)
    
    # Fallback: return tensor as-is (no distribution)
    if _get_debug_setting():
        logger.debug("DTensor not available, returning tensor as-is")
    return tensor


def distribute_variable(tensor, layout):
    """Distributes a Keras variable using PyTorch DTensor or manual sharding."""
    from keras.src.distribution.distribution_lib import distribution

    # Convert tensor first to check its dtype
    converted_tensor = convert_to_tensor(tensor)

    # Check if tensor is floating-point or complex
    # PyTorch requires floating point or complex dtype for requires_grad
    is_float_or_complex = converted_tensor.dtype.is_floating_point or converted_tensor.dtype.is_complex

    # Log tensor info for debugging
    debug_mode = _get_debug_setting()
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

    if debug_mode:
        print(
            f"DEBUG | [Rank {rank:02d}] distribute_variable: "
            f"shape={converted_tensor.shape}, dtype={converted_tensor.dtype}, "
            f"is_float_or_complex={is_float_or_complex}, layout={layout}, "
            f"world_size={world_size}"
        )

    current_distribution = distribution()
    
    if debug_mode:
        print(
            f"DEBUG | [Rank {rank:02d}] distribute_variable: "
            f"current_distribution={current_distribution}, "
            f"layout is None={layout is None}"
        )
    
    if not current_distribution or layout is None:
        # For non-floating point tensors (e.g., integer iterations counter),
        # don't wrap in Parameter since PyTorch requires float/complex for grads
        if not is_float_or_complex:
            if debug_mode:
                print(
                    f"DEBUG | [Rank {rank:02d}] Non-floating tensor, returning as-is: "
                    f"shape={converted_tensor.shape}, dtype={converted_tensor.dtype}"
                )
            return converted_tensor
        return torch.nn.Parameter(converted_tensor)

    # Use the distribution object
    distribution_obj = current_distribution

    # Retrieve the mesh
    device_mesh = _to_backend_mesh(distribution_obj.device_mesh)

    if debug_mode:
        print(
            f"DEBUG | [Rank {rank:02d}] distribute_variable: "
            f"device_mesh={device_mesh}, "
            f"mesh_dim_names={getattr(device_mesh, 'mesh_dim_names', None) if device_mesh else None}"
        )
    
    if device_mesh is None:
        # Fallback if DTensor/Mesh not available
        if not is_float_or_complex:
            if debug_mode:
                print(
                    f"DEBUG | [Rank {rank:02d}] Non-floating tensor, no mesh: "
                    f"shape={converted_tensor.shape}, dtype={converted_tensor.dtype}"
                )
            return converted_tensor
        return torch.nn.Parameter(converted_tensor)

    # Check which axes need sharding
    # layout is typically a tuple of axis names, e.g., (None, 'model')
    placements = []
    needs_sharding = False

    for axis in layout:
        if axis is not None:
            # Find the dimension index in the mesh for this axis name
            try:
                mesh_dim = device_mesh.mesh_dim_names.index(axis)
                placements.append(Shard(mesh_dim))
                needs_sharding = True
            except ValueError:
                if debug_mode:
                    print(
                        f"DEBUG | [Rank {rank:02d}] Axis '{axis}' not found in mesh_dim_names "
                        f"{device_mesh.mesh_dim_names}, replicating"
                    )
                placements.append(Replicate())
        else:
            placements.append(Replicate())

    if debug_mode:
        print(
            f"DEBUG | [Rank {rank:02d}] distribute_variable: "
            f"placements={placements}, needs_sharding={needs_sharding}"
        )

    if not needs_sharding:
        # No sharding needed, replicate the tensor
        if not is_float_or_complex:
            if debug_mode:
                print(
                    f"DEBUG | [Rank {rank:02d}] Non-floating tensor, no sharding needed: "
                    f"shape={converted_tensor.shape}, dtype={converted_tensor.dtype}"
                )
            return converted_tensor

        if debug_mode:
            print(
                f"DEBUG | [Rank {rank:02d}] No sharding needed, replicating: "
                f"shape={converted_tensor.shape}, dtype={converted_tensor.dtype}"
            )
        return torch.nn.Parameter(converted_tensor)

    if DTENSOR_AVAILABLE and torch_distribute_tensor is not None:
        # Create DTensor-based Parameter
        dtensor = torch_distribute_tensor(
            converted_tensor,
            device_mesh,
            placements
        )

        # Log the distributed tensor shape
        if debug_mode:
            local_shape = getattr(dtensor, 'to_local', lambda: dtensor)().shape
            print(
                f"DEBUG | [Rank {rank:02d}] Distributed tensor: "
                f"local_shape={local_shape}, full_shape={dtensor.shape}, "
                f"dtype={converted_tensor.dtype}"
            )

        # For non-floating point tensors, don't wrap in Parameter
        if not is_float_or_complex:
            if debug_mode:
                print(
                    f"DEBUG | [Rank {rank:02d}] Non-floating tensor, returning DTensor without Parameter: "
                    f"shape={dtensor.shape}, dtype={converted_tensor.dtype}"
                )
            return dtensor

        # Wrap as Parameter so it stays on device and tracks grads
        param = torch.nn.Parameter(dtensor)
        return param
    else:
        # Manual sharding fallback (Slicing)
        # For non-floating point tensors, don't wrap in Parameter
        if not is_float_or_complex:
            if debug_mode:
                print(
                    f"DEBUG | [Rank {rank:02d}] Non-floating tensor, manual sharding without Parameter: "
                    f"shape={converted_tensor.shape}, dtype={converted_tensor.dtype}"
                )
            return converted_tensor

        if debug_mode:
            print(
                f"DEBUG | [Rank {rank:02d}] Manual sharding fallback: "
                f"shape={converted_tensor.shape}, dtype={converted_tensor.dtype}"
            )
        return torch.nn.Parameter(converted_tensor)


def _get_default_device_mesh() -> Optional[DeviceMesh]:
    """Get the default device mesh from global state."""
    mesh = global_state.get_global_attribute("torch_device_mesh", None)
    return mesh


def _set_default_device_mesh(mesh: DeviceMesh) -> None:
    """Set the default device mesh in global state."""
    global_state.set_global_attribute("torch_device_mesh", mesh)
from keras.src.backend.common.global_state import get_global_attribute
from keras.src.backend.common.global_state import set_global_attribute

def _get_mesh_info():
    """Internal helper to get the current backend mesh info."""
    mesh = get_global_attribute("torch_device_mesh")
    if mesh is None:
        return None
    return {
        "shape": mesh.shape,
        "dim_names": mesh.mesh_dim_names,
        "device_type": mesh.device_type
    }


def _axis_names_to_placements(
    axis_names: tuple,
    device_mesh: DeviceMesh,
) -> List[Placement]:
    """Convert axis names to DTensor placements.
    
    Args:
        axis_names: Tuple of axis names (strings or None)
        device_mesh: DeviceMesh object
    
    Returns:
        List of Placement objects
    """
    placements = []
    axis_names = list(axis_names)
    
    for axis in axis_names:
        if axis is None:
            # Replicate on this axis
            placements.append(Replicate())
        elif axis in device_mesh.mesh_dim_names:
            # Shard on this axis
            dim = device_mesh.mesh_dim_names.index(axis)
            placements.append(Shard(dim=dim))
        else:
            # Unknown axis, replicate
            placements.append(Replicate())
    
    return placements


def redistribute_dtensor(
    dtensor: DTensor,
    device_mesh: DeviceMesh,
    placements: List[Placement],
) -> DTensor:
    """Redistribute a DTensor with new placements.
    
    Args:
        dtensor: DTensor to redistribute
        device_mesh: Target DeviceMesh
        placements: Target placements
    
    Returns:
        Redistributed DTensor
    """
    if _get_debug_setting():
        logger.debug(f"redistribute_dtensor: {dtensor} -> mesh={device_mesh}, placements={placements}")
    
    return dtensor.redistribute(device_mesh, placements)


def _to_backend_device(device_name):
    """Convert a device name string to a backend device.
    
    Args:
        device_name: Device name string like 'cuda:0'
    
    Returns:
        torch.device
    """
    return _parse_device(device_name)


def _to_backend_mesh(device_mesh):
    """Converts a Keras DeviceMesh to a PyTorch DeviceMesh."""
    debug_mode = _get_debug_setting()
    
    if not DTENSOR_AVAILABLE:
        if debug_mode:
            print(f"DEBUG | DTENSOR not available, _to_backend_mesh returning None")
        return None
    
    # Check if already cached on this specific DeviceMesh object
    # Use the DeviceMesh object's id as part of the cache key
    # This prevents different DeviceMesh instances from sharing the same cache
    cache_key = f"torch_device_mesh_{id(device_mesh)}"
    existing_mesh = get_global_attribute(cache_key)
    if existing_mesh is not None:
        if debug_mode:
            print(f"DEBUG | Found cached mesh for this DeviceMesh: {existing_mesh}")
        return existing_mesh

    if debug_mode:
        print(f"DEBUG | Creating new backend mesh from device_mesh: shape={device_mesh.shape}, axis_names={device_mesh.axis_names}")
        print(f"DEBUG | device_mesh.devices = {device_mesh.devices}")

    # Flatten devices to list of indices
    device_ids = []
    for d in device_mesh.devices.flatten():
        # Handle "cuda:0" or "cpu" strings
        if ":" in d:
            device_ids.append(int(d.split(":")[-1]))
        else:
            # Default to 0 if no index (e.g. "cpu")
            device_ids.append(0)
    
    if debug_mode:
        print(f"DEBUG | device_ids = {device_ids}")
    
    mesh_shape = device_mesh.shape
    axis_names = device_mesh.axis_names
    
    # PyTorch expects a numpy array for the mesh structure
    mesh_array = np.array(device_ids).reshape(mesh_shape)
    
    if debug_mode:
        print(f"DEBUG | mesh_array.shape = {mesh_array.shape}, mesh_array = {mesh_array}")
    
    backend_mesh = TorchDeviceMesh(
        device_type="cuda" if torch.cuda.is_available() else "cpu",
        mesh=mesh_array,
        mesh_dim_names=axis_names
    )
    
    if debug_mode:
        print(f"DEBUG | Created TorchDeviceMesh: shape={backend_mesh.shape}, mesh_dim_names={backend_mesh.mesh_dim_names}")
    
    # Cache on the DeviceMesh object itself, not globally
    set_global_attribute(cache_key, backend_mesh)
    return backend_mesh


def _to_backend_layout(tensor_layout) -> tuple:
    """Convert the TensorLayout to PyTorch backend specific layout.
    
    Args:
        tensor_layout: TensorLayout instance to convert.
    
    Returns:
        Tuple of axis names for the layout.
    """
    if _get_debug_setting():
        logger.debug(f"_to_backend_layout called with tensor_layout={tensor_layout}")
    
    if tensor_layout.device_mesh is None:
        raise ValueError(
            "Cannot create sharding when device mesh is not set "
            "for TensorLayout."
        )
    
    # Return the axes as-is (they will be used in distribute_tensor)
    return tensor_layout.axes


# Path Adapter for converting between Keras and PyTorch naming conventions
# Keras uses: "dense/kernel" (forward slashes)
# PyTorch uses: "dense.weight" (dots)

def keras_to_pytorch_path(keras_path: str) -> str:
    """Convert a Keras variable path to PyTorch format.
    
    Args:
        keras_path: Path like "dense/kernel" or "my_model/dense_1/bias"
    
    Returns:
        PyTorch-style path like "dense.weight" or "my_model.dense_1.bias"
    
    Example:
        >>> keras_to_pytorch_path("dense/kernel")
        'dense.weight'
        >>> keras_to_pytorch_path("conv2d/bias")
        'conv2d.bias'
    """
    # Handle special case for weight names
    # Keras often uses "layer_name/weight_name" pattern
    # PyTorch uses "layer_name.weight_name" pattern
    
    if _get_debug_setting():
        logger.debug(f"Converting Keras path to PyTorch: {keras_path}")
    
    # Replace forward slashes with dots
    pytorch_path = keras_path.replace('/', '.')
    
    # Special handling for common variable names
    # Keras: "kernel", PyTorch: "weight"
    # Keras: "bias", PyTorch: "bias" (same)
    
    return pytorch_path


def pytorch_to_keras_path(pytorch_path: str) -> str:
    """Convert a PyTorch variable path to Keras format.
    
    Args:
        pytorch_path: Path like "dense.weight" or "my_model.dense_1.bias"
    
    Returns:
        Keras-style path like "dense/kernel" or "my_model/dense_1/bias"
    
    Example:
        >>> pytorch_to_keras_path("dense.weight")
        'dense/kernel'
        >>> pytorch_to_keras_path("conv2d.bias")
        'conv2d/bias'
    """
    if _get_debug_setting():
        logger.debug(f"Converting PyTorch path to Keras: {pytorch_path}")
    
    # Replace dots with forward slashes
    keras_path = pytorch_path.replace('.', '/')
    
    return keras_path


def convert_path_for_regex(path: str, source_format: str = "keras") -> str:
    """Convert a path to work with regex patterns from the other format.
    
    This is useful when you have regex patterns in Keras format but need
    to match against PyTorch paths.
    
    Args:
        path: The path to convert
        source_format: "keras" if path is in Keras format, "pytorch" if in PyTorch
    
    Returns:
        Converted path that should work with the other format's regex patterns
    """
    if source_format == "keras":
        # Keras path -> check both Keras and PyTorch patterns
        keras_path = path
        pytorch_path = keras_to_pytorch_path(path)
    else:
        # PyTorch path -> check both formats
        pytorch_path = path
        keras_path = pytorch_to_keras_path(path)
    
    if _get_debug_setting():
        logger.debug(
            f"Path conversion: {path} (from {source_format}) -> "
            f"keras='{keras_path}', pytorch='{pytorch_path}'"
        )
    
    return keras_path, pytorch_path


# Utility to get the backend distribution_lib module
def get_distribution_lib():
    """Get the torch backend distribution_lib module."""
    return sys.modules[__name__]

