"""Distribution strategy utilities for PyTorch backend using DTensor.

This module provides distribution support for PyTorch backend using PyTorch's
DTensor API for distributed tensor computing.
"""

import os
from typing import List, Optional

import numpy as np
import torch

from keras.src.backend.common import global_state

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
from torch.distributed.tensor.parallel import (
        parallelize_module,
        ColwiseParallel,
        RowwiseParallel,
    )
TENSOR_PARALLEL_AVAILABLE = True

# Global state tracking
_DISTRIBUTION_INITIALIZED = False


def list_devices(device_type: Optional[str] = None) -> List[str]:
    """Return all available devices based on device type."""
    device_type = device_type.lower() if device_type else None
    devices = []

    # Check TPU
    if device_type in (None, 'tpu'):
        try:
            import torch_xla.core.xla_model as xm
            tpu_devices = xm.get_xla_supported_devices('tpu')
            devices.extend([f'tpu:{i}' for i in range(len(tpu_devices))])
            if device_type == 'tpu':
                return devices
        except ImportError:
            pass

    # Check GPU/CUDA
    if device_type in (None, 'gpu', 'cuda'):
        if torch.cuda.is_available():
            devices.extend([f'cuda:{i}' for i in range(torch.cuda.device_count())])
            if device_type in ('gpu', 'cuda'):
                return devices

    # Check CPU
    if device_type in (None, 'cpu'):
        num_cpu = os.cpu_count() or 4
        devices.extend([f'cpu:{i}' for i in range(num_cpu)])
        return devices

    # Return non-CPU devices if no specific type requested
    if device_type is None:
        non_cpu = [d for d in devices if not d.startswith('cpu:')]
        return non_cpu if non_cpu else devices

    return devices


def get_device_count(device_type: Optional[str] = None) -> int:
    """Returns the number of available devices."""
    device_type = device_type.lower() if device_type else None

    if device_type in (None, 'tpu'):
        try:
            import torch_xla.core.xla_model as xm
            return len(xm.get_xla_supported_devices('tpu'))
        except ImportError:
            pass

    if device_type in (None, 'gpu', 'cuda'):
        if torch.cuda.is_available():
            return torch.cuda.device_count()

    if device_type in (None, 'cpu'):
        return os.cpu_count() or 1

    return 0


def initialize(
    job_addresses: Optional[str] = None,
    num_processes: Optional[int] = None,
    process_id: Optional[int] = None,
) -> None:
    """Initialize the distribution system for multi-process settings."""
    global _DISTRIBUTION_INITIALIZED

    # Check torchrun environment variables first
    local_rank = os.environ.get("LOCAL_RANK")
    world_size_env = os.environ.get("WORLD_SIZE")

    if local_rank is not None and world_size_env is not None:
        num_processes = num_processes or int(world_size_env)
        process_id = process_id or int(local_rank)

    # Check Keras environment variables
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

    # Single process or torchrun already initialized
    if not num_processes or num_processes == 1 or torch.distributed.is_initialized():
        _DISTRIBUTION_INITIALIZED = True
        return

    # Multi-process initialization
    if job_addresses and "," in job_addresses:
        init_method = f"tcp://{job_addresses.split(',')[0]}"
    elif job_addresses:
        init_method = f"tcp://{job_addresses}"
    else:
        init_method = "env://"

    torch.distributed.init_process_group(
        backend="nccl" if torch.cuda.is_available() else "gloo",
        init_method=init_method,
        world_size=num_processes,
        rank=process_id or 0,
    )

    _DISTRIBUTION_INITIALIZED = True


def num_processes() -> int:
    """Return the number of processes for the current distribution setting."""
    return torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1


def process_id() -> int:
    """Return the current process ID for the distribution setting."""
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0


def distribute_tensor(tensor: torch.Tensor, layout) -> torch.Tensor:
    """Distribute the tensor based on the layout."""
    if layout is None:
        if tensor.requires_grad:
            return tensor
        if tensor.dtype.is_floating_point:
            return tensor.requires_grad_(True)
        return tensor

    # Get backend layout
    backend_layout = getattr(layout, 'backend_layout', layout)

    # Handle DTensor sharding
    if DTENSOR_AVAILABLE:
        device_mesh = _get_default_device_mesh()
        if device_mesh is not None:
            if hasattr(backend_layout, 'placements'):
                if isinstance(tensor, DTensor):
                    return tensor.redistribute(device_mesh, backend_layout.placements)
                return torch_distribute_tensor(tensor, device_mesh, backend_layout.placements)
            elif isinstance(backend_layout, tuple):
                placements = _axis_names_to_placements(backend_layout, device_mesh)
                if isinstance(tensor, DTensor):
                    return tensor.redistribute(device_mesh, placements)
                return torch_distribute_tensor(tensor, device_mesh, placements)

    return tensor


def distribute_variable(tensor, layout=None, module_name=None):
    """Distributes a Keras variable using PyTorch DTensor."""
    from keras.src.distribution.distribution_lib import distribution

    converted_tensor = convert_to_tensor(tensor)
    is_float_or_complex = converted_tensor.dtype.is_floating_point or converted_tensor.dtype.is_complex

    # Check if ModelParallel distribution is active
    current_distribution = distribution()
    is_model_parallel = current_distribution is not None

    # For ModelParallel with sharding layout
    if is_model_parallel and layout is not None and DTENSOR_AVAILABLE:
        device_mesh = _to_backend_mesh(current_distribution.device_mesh)
        if device_mesh is not None:
            placements = _layout_to_placements(layout, converted_tensor, device_mesh)
            if any(isinstance(p, Shard) for p in placements):
                dtensor = torch_distribute_tensor(converted_tensor, device_mesh, placements)
                return torch.nn.Parameter(dtensor) if is_float_or_complex else dtensor

    # Default: return as Parameter for gradient tracking
    return torch.nn.Parameter(converted_tensor) if is_float_or_complex else converted_tensor


def _layout_to_placements(layout, tensor, device_mesh):
    """Convert Keras layout tuple to DTensor placements."""
    placements = []
    tensor_rank = tensor.dim()
    mesh_ndim = len(device_mesh.mesh_dim_names)

    for i, axis in enumerate(layout):
        if axis is not None:
            try:
                mesh_dim = device_mesh.mesh_dim_names.index(axis)
                tensor_dim = tensor_rank - len(layout) + i if tensor_rank > len(layout) else i
                if tensor_rank == 1:
                    tensor_dim = 0
                placements.append(Shard(tensor_dim))
            except ValueError:
                placements.append(Replicate())
        else:
            placements.append(Replicate())

    # Pad placements to match mesh dimensions
    while len(placements) < mesh_ndim:
        placements.append(Replicate())

    return placements


def _get_default_device_mesh() -> Optional[DeviceMesh]:
    """Get the default device mesh from global state."""
    return global_state.get_global_attribute("torch_device_mesh", None)


def _axis_names_to_placements(axis_names, device_mesh):
    """Convert axis names to DTensor placements."""
    return [
        Replicate() if axis is None else Shard(device_mesh.mesh_dim_names.index(axis))
        for axis in axis_names
    ]


def _to_backend_mesh(device_mesh):
    """Converts a Keras DeviceMesh to a PyTorch DeviceMesh."""
    if not DTENSOR_AVAILABLE:
        return None

    cache_key = f"torch_mesh_{device_mesh.shape}_{device_mesh.axis_names}"
    cached = global_state.get_global_attribute(cache_key)
    if cached is not None:
        global_state.set_global_attribute("torch_device_mesh", cached)
        return cached

    device_ids = [int(d.split(":")[-1]) if ":" in d else 0 for d in device_mesh.devices.flatten()]

    backend_mesh = TorchDeviceMesh(
        device_type="cuda" if torch.cuda.is_available() else "cpu",
        mesh=np.array(device_ids).reshape(device_mesh.shape),
        mesh_dim_names=device_mesh.axis_names
    )

    global_state.set_global_attribute(cache_key, backend_mesh)
    global_state.set_global_attribute("torch_device_mesh", backend_mesh)
    return backend_mesh


def _to_backend_layout(tensor_layout):
    """Convert TensorLayout to backend layout tuple."""
    if tensor_layout.device_mesh is None:
        raise ValueError("Cannot create sharding without device mesh")
    return tensor_layout.axes


def parallelize_torch_module(module, device_mesh, layout_map):
    """Parallelize a PyTorch module using tensor parallelism."""
    if not TENSOR_PARALLEL_AVAILABLE:
        raise ImportError("PyTorch tensor.parallel is not available")
    if device_mesh is None:
        raise ValueError("device_mesh cannot be None")

    return parallelize_module(module, device_mesh, parallel_plan=layout_map)


def create_tp_plan_from_layout_map(module, keras_layout_map):
    """Create tensor parallel plan from Keras-style layout map."""
    if not keras_layout_map or not TENSOR_PARALLEL_AVAILABLE:
        return {}

    styles = {0: RowwiseParallel(), 1: ColwiseParallel()}
    plan = {}

    for pattern, sharding_spec in keras_layout_map.items():
        if sharding_spec is None:
            continue

        # Extract axes from TensorLayout if needed
        if hasattr(sharding_spec, 'axes'):
            sharding_spec = sharding_spec.axes

        # Convert path format
        pytorch_pattern = pattern.replace('/', '.')

        if isinstance(sharding_spec, tuple):
            model_idx = next((i for i, axis in enumerate(sharding_spec) if axis == 'model'), None)
            if model_idx is not None:
                plan[pytorch_pattern] = styles.get(model_idx, ColwiseParallel())
        elif isinstance(sharding_spec, str) and sharding_spec == 'model':
            plan[pytorch_pattern] = ColwiseParallel()

    return plan


def _to_dtensor(tensor, device_mesh=None, placements=None):
    """Convert a tensor to DTensor if it isn't already."""
    if not DTENSOR_AVAILABLE or tensor is None or isinstance(tensor, DTensor):
        return tensor

    device_mesh = device_mesh or _get_default_device_mesh()
    if device_mesh is None:
        return tensor

    placements = [Replicate()] if placements is None else (placements if isinstance(placements, list) else [placements])
    return DTensor.from_local(tensor, device_mesh, placements)


def is_dtensor(tensor):
    """Check if a tensor is a DTensor."""
    return DTENSOR_AVAILABLE and isinstance(tensor, DTensor)


def dtensor_to_local(tensor):
    """Convert DTensor to local tensor format."""
    if tensor is None:
        return tensor
    if isinstance(tensor, DTensor):
        return tensor.to_local()
    if isinstance(tensor, dict):
        return {k: dtensor_to_local(v) for k, v in tensor.items()}
    if isinstance(tensor, (list, tuple)):
        return type(tensor)(dtensor_to_local(v) for v in tensor)
    return tensor


from keras.src.distribution.path_utils import (
    keras_to_pytorch_path,
    pytorch_to_keras_path,
    convert_path_for_matching,
)

