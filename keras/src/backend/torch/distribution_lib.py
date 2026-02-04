"""Distribution strategy utilities for PyTorch backend using DTensor.

This module provides distribution support for PyTorch backend using PyTorch's
DTensor API for distributed tensor computing.
"""

import os

import torch

from keras.src.backend.common import global_state
from keras.src.backend.torch.core import convert_to_tensor

from torch.distributed._tensor import DTensor, DeviceMesh, Replicate, Shard
from torch.distributed._tensor.api import distribute_tensor as torch_distribute_tensor
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel

TENSOR_PARALLEL_AVAILABLE = True


def list_devices(device_type=None):
    """Return all available devices based on device type."""
    device_type = device_type.lower() if device_type else None
    devices = []
    check_types = [device_type] if device_type else ['tpu', 'gpu', 'cpu']

    for dtype in check_types:
        if dtype == 'tpu':
            import torch_xla.core.xla_model as xm
            tpu_devices = xm.get_xla_supported_devices('tpu')
            devices.extend([f'tpu:{i}' for i in range(len(tpu_devices))])
            if device_type == 'tpu':
                return devices
        elif dtype in ('gpu', 'cuda'):
            if torch.cuda.is_available():
                devices.extend([f'cuda:{i}' for i in range(torch.cuda.device_count())])
                if device_type in ('gpu', 'cuda'):
                    return devices
        elif dtype == 'cpu':
            devices.extend([f'cpu:{i}' for i in range(os.cpu_count() or 4)])
            return devices

    return [d for d in devices if not d.startswith('cpu:')] or devices if device_type is None else []


def get_device_count(device_type=None):
    """Returns the number of available devices."""
    device_type = device_type.lower() if device_type else None

    if device_type in (None, 'tpu'):
        import torch_xla.core.xla_model as xm
        return len(xm.get_xla_supported_devices('tpu'))

    if device_type in (None, 'gpu', 'cuda'):
        if torch.cuda.is_available():
            return torch.cuda.device_count()

    if device_type in (None, 'cpu'):
        return os.cpu_count() or 1

    return 0


def initialize(job_addresses=None, num_processes=None, process_id=None):
    """Initialize the distribution system for multi-process settings."""
    local_rank = os.environ.get("LOCAL_RANK")
    world_size_env = os.environ.get("WORLD_SIZE")

    if local_rank is not None and world_size_env is not None:
        num_processes = num_processes or int(world_size_env)
        process_id = process_id or int(local_rank)

    env_map = {
        "KERAS_DISTRIBUTION_JOB_ADDRESSES": "job_addresses",
        "KERAS_DISTRIBUTION_NUM_PROCESSES": "num_processes",
        "KERAS_DISTRIBUTION_PROCESS_ID": "process_id"
    }

    for env_var, param in env_map.items():
        if locals()[param] is None:
            locals()[param] = os.environ.get(env_var)

    if isinstance(num_processes, str):
        num_processes = int(num_processes)
    if isinstance(process_id, str):
        process_id = int(process_id)

    if not num_processes or num_processes == 1 or torch.distributed.is_initialized():
        return

    # Set the CUDA device for this rank (required for NCCL to work correctly)
    if torch.cuda.is_available() and local_rank is not None:
        torch.cuda.set_device(int(local_rank))

    init_method = f"tcp://{job_addresses}" if job_addresses else "env://"
    if "," in str(job_addresses):
        init_method = f"tcp://{job_addresses.split(',')[0]}"

    torch.distributed.init_process_group(
        backend="nccl" if torch.cuda.is_available() else "gloo",
        init_method=init_method,
        world_size=num_processes,
        rank=process_id or 0,
    )


def num_processes():
    """Return the number of processes for the current distribution setting."""
    return torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1


def process_id():
    """Return the current process ID for the distribution setting."""
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0


def distribute_tensor(tensor, layout):
    """Distribute the tensor based on the layout."""
    if layout is None:
        return tensor if not tensor.requires_grad else tensor.requires_grad_(True)

    backend_layout = getattr(layout, 'backend_layout', layout)
    device_mesh = _get_default_device_mesh()
    if device_mesh is None:
        return tensor

    placements = (backend_layout.placements if hasattr(backend_layout, 'placements')
                 else _axis_names_to_placements(backend_layout, device_mesh) if isinstance(backend_layout, tuple) else None)
    if placements:
        return tensor.redistribute(device_mesh, placements) if isinstance(tensor, DTensor) else torch_distribute_tensor(tensor, device_mesh, placements)

    return tensor


def distribute_variable(tensor, layout=None):
    """Distributes a Keras variable using PyTorch DTensor."""
    from keras.src.distribution.distribution_lib import distribution

    converted_tensor = convert_to_tensor(tensor)
    is_float_or_complex = converted_tensor.dtype.is_floating_point or converted_tensor.dtype.is_complex

    current_distribution = distribution()
    if current_distribution is not None:
        device_mesh = _to_backend_mesh(current_distribution.device_mesh)
        if device_mesh is not None:
            placements = _layout_to_placements(layout, converted_tensor, device_mesh) if layout else None
            if placements and any(isinstance(p, Shard) for p in placements):
                dtensor = torch_distribute_tensor(converted_tensor, device_mesh, placements)
                return torch.nn.Parameter(dtensor) if is_float_or_complex else dtensor

    return torch.nn.Parameter(converted_tensor) if is_float_or_complex else converted_tensor


def _layout_to_placements(layout, tensor, device_mesh):
    """Convert Keras layout tuple to DTensor placements."""
    placements = []
    tensor_rank = tensor.dim()
    mesh_ndim = len(device_mesh.mesh_dim_names)

    for i, axis in enumerate(layout):
        if axis is not None:
            mesh_dim = device_mesh.mesh_dim_names.index(axis)
            tensor_dim = tensor_rank - len(layout) + i if tensor_rank > len(layout) else (0 if tensor_rank == 1 else i)
            placements.append(Shard(tensor_dim))
        else:
            placements.append(Replicate())

    while len(placements) < mesh_ndim:
        placements.append(Replicate())

    return placements


def _get_default_device_mesh():
    """Get the default device mesh from global state."""
    return global_state.get_global_attribute("torch_device_mesh", None)


def _axis_names_to_placements(axis_names, device_mesh):
    """Convert axis names to DTensor placements."""
    return [Replicate() if axis is None else Shard(device_mesh.mesh_dim_names.index(axis)) for axis in axis_names]


def _to_backend_mesh(device_mesh):
    """Converts a Keras DeviceMesh to a PyTorch DeviceMesh."""
    cache_key = f"torch_mesh_{device_mesh.shape}_{device_mesh.axis_names}"
    cached = global_state.get_global_attribute(cache_key)
    if cached is not None:
        global_state.set_global_attribute("torch_device_mesh", cached)
        return cached

    device_ids = [int(d.split(":")[-1]) if ":" in d else 0 for d in device_mesh.devices.flatten()]
    backend_mesh = DeviceMesh(
        device_type="cuda" if torch.cuda.is_available() else "cpu",
        mesh=torch.tensor(device_ids).reshape(device_mesh.shape),
        mesh_dim_names=device_mesh.axis_names
    )

    global_state.set_global_attribute(cache_key, backend_mesh)
    global_state.set_global_attribute("torch_device_mesh", backend_mesh)
    return backend_mesh


def _to_backend_layout(tensor_layout):
    """Convert TensorLayout to backend layout tuple."""
    from keras.src.distribution.distribution_lib import TensorLayout

    if tensor_layout.device_mesh is None:
        raise ValueError("Cannot create sharding without device mesh")
    return tensor_layout.axes


def _get_tp_mesh_from_2d_mesh(device_mesh):
    """Extract 1D tensor parallel mesh from 2D DeviceMesh.

    PyTorch's parallelize_module API only accepts a 1D DeviceMesh.
    When we have a 2D mesh with shape (batch, model), we need to
    extract the "model" axis for tensor parallel operations.

    Args:
        device_mesh: A DeviceMesh instance (can be 1D or 2D)

    Returns:
        A 1D DeviceMesh suitable for tensor parallelism
    """
    if device_mesh.mesh.ndim == 1:
        return device_mesh

    # For 2D mesh, find the 'model' axis and extract it
    mesh_dim_names = device_mesh.mesh_dim_names
    if 'model' in mesh_dim_names:
        model_dim = mesh_dim_names.index('model')
        # Extract the model axis mesh
        # Create indices to select only the model dimension
        indices = [slice(None)] * device_mesh.mesh.ndim
        indices[model_dim] = slice(None)  # Select all indices along model axis
        model_mesh_tensor = device_mesh.mesh[tuple(indices)]

        # Handle the case where the mesh tensor might be 2D even for single index
        if model_mesh_tensor.ndim > 1:
            # Flatten if needed
            model_mesh_tensor = model_mesh_tensor.reshape(-1)

        tp_mesh = DeviceMesh(
            device_type=device_mesh.device_type,
            mesh=model_mesh_tensor,
            mesh_dim_names=['model']
        )
        return tp_mesh

    # Fallback: use the last dimension
    last_dim = device_mesh.mesh.ndim - 1
    indices = [slice(None)] * device_mesh.mesh.ndim
    indices[last_dim] = slice(None)
    last_mesh_tensor = device_mesh.mesh[tuple(indices)]
    if last_mesh_tensor.ndim > 1:
        last_mesh_tensor = last_mesh_tensor.reshape(-1)

    return DeviceMesh(
        device_type=device_mesh.device_type,
        mesh=last_mesh_tensor,
        mesh_dim_names=[mesh_dim_names[last_dim]]
    )


def parallelize_torch_module(module, device_mesh, layout_map):
    """Parallelize a PyTorch module using tensor parallelism."""
    if device_mesh is None:
        raise ValueError("device_mesh cannot be None")

    from keras.src.distribution.distribution_lib import LayoutMap
    if isinstance(layout_map, LayoutMap):
        layout_map = create_tp_plan_from_layout_map(module, dict(layout_map))

    # PyTorch's parallelize_module only accepts a 1D DeviceMesh
    # For 2D meshes, we need to extract the tensor parallel dimension
    if hasattr(device_mesh, 'mesh') and device_mesh.mesh.ndim > 1:
        tp_mesh = _get_tp_mesh_from_2d_mesh(device_mesh)
        return parallelize_module(module, tp_mesh, parallelize_plan=layout_map)

    return parallelize_module(module, device_mesh, parallelize_plan=layout_map)


def create_tp_plan_from_layout_map(module, keras_layout_map):
    """Create tensor parallel plan from Keras-style layout map."""
    if not keras_layout_map:
        return {}

    styles = {0: RowwiseParallel(), 1: ColwiseParallel()}
    plan = {}

    for pattern, sharding_spec in keras_layout_map.items():
        if sharding_spec is None:
            continue

        if hasattr(sharding_spec, 'axes'):
            sharding_spec = sharding_spec.axes

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
    if tensor is None or isinstance(tensor, DTensor):
        return tensor

    device_mesh = device_mesh or _get_default_device_mesh()
    if device_mesh is None:
        return tensor

    placements = [Replicate()] if placements is None else (placements if isinstance(placements, list) else [placements])
    return DTensor.from_local(tensor, device_mesh, placements)


def is_dtensor(tensor):
    """Check if a tensor is a DTensor."""
    return isinstance(tensor, DTensor)


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


get_dtensor_local = dtensor_to_local


class _AllGatherWithGradient(torch.autograd.Function):
    """Custom autograd function for all-gather with proper gradient flow."""

    @staticmethod
    def forward(ctx, local_tensor, shard_dim):
        world_size = torch.distributed.get_world_size()
        output_shape = list(local_tensor.shape)
        output_shape[shard_dim] *= world_size

        output = torch.empty(output_shape, dtype=local_tensor.dtype, device=local_tensor.device)

        if hasattr(torch.distributed, "all_gather_into_tensor"):
            torch.distributed.all_gather_into_tensor(output, local_tensor.contiguous())
        else:
            output_list = [torch.empty_like(local_tensor) for _ in range(world_size)]
            torch.distributed.all_gather(output_list, local_tensor.contiguous())
            output = torch.cat(output_list, dim=shard_dim)

        ctx.shard_dim = shard_dim
        ctx.world_size = world_size
        return output

    @staticmethod
    def backward(ctx, grad_output):
        shard_dim = ctx.shard_dim
        world_size = ctx.world_size
        rank = torch.distributed.get_rank()

        shard_size = grad_output.shape[shard_dim] // world_size
        grad_local = grad_output[tuple(slice(None) if i != shard_dim else slice(rank * shard_size, (rank + 1) * shard_size) for i in range(grad_output.dim()))]

        return grad_local.contiguous(), None


def _all_gather_with_grad(local_tensor, shard_dim):
    """Perform all-gather with proper gradient flow."""
    return _AllGatherWithGradient.apply(local_tensor, shard_dim)


def _convert_structure(x, device_mesh=None, to_dtensor=True, gather_sharded=True):
    """Unified recursive structure converter for DTensor operations."""
    if x is None:
        return x

    if isinstance(x, DTensor):
        if not to_dtensor:
            if gather_sharded and not all(isinstance(p, Replicate) for p in x.placements):
                if torch.distributed.is_initialized():
                    shard_dim = next((i for i, p in enumerate(x.placements) if isinstance(p, Shard)), None)
                    if shard_dim is not None:
                        local_tensor = x.to_local()
                        if local_tensor.requires_grad:
                            return _all_gather_with_grad(local_tensor, shard_dim)
                        else:
                            output = [torch.empty_like(local_tensor) for _ in range(torch.distributed.get_world_size())]
                            torch.distributed.all_gather(output, local_tensor.contiguous())
                            return torch.cat(output, dim=shard_dim)
            return x.to_local()
        return x

    if isinstance(x, torch.Tensor):
        return DTensor.from_local(x, device_mesh, [Replicate()]) if to_dtensor else x

    if isinstance(x, dict):
        return {k: _convert_structure(v, device_mesh, to_dtensor, gather_sharded) for k, v in x.items()}

    if isinstance(x, (list, tuple)):
        return type(x)(_convert_structure(v, device_mesh, to_dtensor, gather_sharded) for v in x)

    return x


def _is_model_parallel_distribution():
    """Check if ModelParallel distribution is active and distributed is initialized."""
    from keras.src.distribution.distribution_lib import distribution, ModelParallel
    dist = distribution()
    return isinstance(dist, ModelParallel) and torch.distributed.is_initialized()


def prepare_input_for_distribution(x):
    """Convert inputs to DTensors when model has DTensor weights."""
    device_mesh = _get_default_device_mesh()
    if device_mesh is None or not _is_model_parallel_distribution():
        return x

    return _convert_structure(x, device_mesh, to_dtensor=True, gather_sharded=False)


def prepare_output_for_loss(x):
    """Convert DTensor outputs to local tensors."""
    if not _is_model_parallel_distribution():
        return x

    return _convert_structure(x, None, to_dtensor=False, gather_sharded=True)

