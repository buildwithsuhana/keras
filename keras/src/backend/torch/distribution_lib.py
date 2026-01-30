"""Utilities for distribution strategy with PyTorch backend using DTensor.

This module provides distribution support for PyTorch backend using PyTorch's
DTensor API. DTensor is a PyTorch API for distributed tensor computing that
provides a unified interface for both data parallelism and model parallelism.

Key features:
- Device mesh creation and management
- Tensor layout specification and distribution
- Support for CPU, GPU, and TPU devices
- Path adapter for converting between Keras `/` paths and PyTorch `.` paths

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

import os
from typing import List, Optional

import numpy as np
import torch

from keras.src.backend.common import global_state

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
from keras.src.backend.torch.core import convert_to_tensor

# Global variable to track if distribution is initialized
_DISTRIBUTION_INITIALIZED = False

# Global variable to track if model has been parallelized
_MODEL_PARALLELIZED = False

# Global variable to track if sharded DTensor weights were created
# When this is True, we should NOT call parallelize_module()
_DISTRIBUTED_WEIGHTS_CREATED = False


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
    
    # Check for CPU
    if device_type in (None, 'cpu'):
        # For CPU, we simulate multiple devices for parallel execution
        num_cpu = os.cpu_count() or 4
        devices.extend([f'cpu:{i}' for i in range(num_cpu)])
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
    device_type = device_type.lower() if device_type else None
    
    # TPU
    if device_type in (None, 'tpu'):
        try:
            import torch_xla.core.xla_model as xm
            tpu_devices = xm.get_xla_supported_devices('tpu')
            return len(tpu_devices)
        except ImportError:
            pass
    
    # GPU/CUDA
    if device_type in (None, 'gpu', 'cuda'):
        if torch.cuda.is_available():
            return torch.cuda.device_count()
    
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
        else:
            _DISTRIBUTION_INITIALIZED = True
        return
    
    # For multi-process, initialize PyTorch distributed
    # First check if torchrun has already initialized it
    if torch.distributed.is_initialized():
        _DISTRIBUTION_INITIALIZED = True
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
    
    # No layout specified and DTensor not available, returning as-is (normal behavior)
    return tensor


def distribute_variable(tensor, layout=None, module_name=None):
    """Distributes a Keras variable using PyTorch DTensor with tensor parallelism.
    
    This function uses tensor parallelism with ColwiseParallel/RowwiseParallel
    styles for proper model parallelism support.
    
    Args:
        tensor: The tensor to distribute
        layout: The layout specification (tuple of axis names or None)
        module_name: Optional name of the module this variable belongs to
                    (for tensor parallel plan creation)
    """
    from keras.src.distribution.distribution_lib import distribution

    # Convert tensor first to check its dtype
    converted_tensor = convert_to_tensor(tensor)

    # Check if tensor is floating-point or complex
    # PyTorch requires floating point or complex dtype for requires_grad
    is_float_or_complex = converted_tensor.dtype.is_floating_point or converted_tensor.dtype.is_complex

    # Check if ModelParallel distribution is active (for tensor parallelism)
    is_model_parallel = False
    current_distribution = distribution()
    if current_distribution is not None:
        from keras.src.distribution.distribution_lib import ModelParallel
        is_model_parallel = isinstance(current_distribution, ModelParallel)
    
    # =====================================================================
    # KEY FIX: For ModelParallel, create SHARDED DTensor directly
    # to avoid OOM - full weights should NEVER exist in memory
    # =====================================================================
    if is_model_parallel and layout is not None:
        # Get device mesh for sharding
        distribution_obj = current_distribution
        device_mesh = _to_backend_mesh(distribution_obj.device_mesh)
        
        if device_mesh is not None and DTENSOR_AVAILABLE:
            # Calculate sharding placements from layout
            placements = []
            needs_sharding = False
            tensor_rank = converted_tensor.dim()
            mesh_ndim = len(device_mesh.mesh_dim_names)
            
            for i, axis in enumerate(layout):
                if axis is not None:
                    try:
                        mesh_dim = device_mesh.mesh_dim_names.index(axis)
                        # For 2D weight (out_dim, in_dim), shard on output dim
                        tensor_dim = tensor_rank - len(layout) + i
                        if tensor_rank == 1:
                            tensor_dim = 0
                        placements.append(Shard(tensor_dim))
                        needs_sharding = True
                    except ValueError:
                        placements.append(Replicate())
                else:
                    placements.append(Replicate())
            
            # Pad placements if needed
            if len(placements) < mesh_ndim:
                placements.extend([Replicate()] * (mesh_ndim - len(placements)))
            
            if needs_sharding and DTENSOR_AVAILABLE:
                # Create DTensor directly - ONLY local shard exists on each device
                # Full tensor NEVER exists in memory
                if torch_distribute_tensor is not None:
                    dtensor = torch_distribute_tensor(
                        converted_tensor,
                        device_mesh,
                        placements
                    )
                    
                    # Mark that distributed weights were created
                    # This prevents parallelize_module from being called
                    global _DISTRIBUTED_WEIGHTS_CREATED
                    _DISTRIBUTED_WEIGHTS_CREATED = True
                    
                    # Return as Parameter for gradient tracking
                    return torch.nn.Parameter(dtensor)
            
            # No sharding needed, replicate
            return torch.nn.Parameter(converted_tensor) if is_float_or_complex else converted_tensor
    
    # If ModelParallel is active but no sharding layout, create regular Parameter
    if is_model_parallel:
        return torch.nn.Parameter(converted_tensor) if is_float_or_complex else converted_tensor
    
    # If no distribution or no layout, return non-distributed parameter
    if not current_distribution or layout is None:
        return torch.nn.Parameter(converted_tensor) if is_float_or_complex else converted_tensor

    # Use the distribution object
    distribution_obj = current_distribution

    # Retrieve the mesh
    device_mesh = _to_backend_mesh(distribution_obj.device_mesh)
    
    if device_mesh is None:
        # No device mesh available, using regular Parameter (replicated)
        return torch.nn.Parameter(converted_tensor) if is_float_or_complex else converted_tensor

    # Check which axes need sharding
    # layout is typically a tuple of axis names, e.g., (None, 'model')
    placements = []
    needs_sharding = False
    
    # Get tensor rank (number of dimensions)
    tensor_rank = converted_tensor.dim()
    mesh_ndim = len(device_mesh.mesh_dim_names)

    for i, axis in enumerate(layout):
        if axis is not None:
            # Find the dimension index in the mesh for this axis name
            try:
                mesh_dim = device_mesh.mesh_dim_names.index(axis)
                
                # Map mesh dimension to tensor dimension
                # For tensors with fewer dimensions than the mesh, we need to adjust
                if tensor_rank <= mesh_ndim:
                    # Map: mesh_dim -> tensor_dim (adjust for shorter tensors)
                    tensor_dim = tensor_rank - mesh_ndim + i
                    if tensor_dim < 0:
                        tensor_dim = i
                else:
                    # For tensors with more dimensions than the mesh
                    tensor_dim = tensor_rank - len(layout) + i
                
                # For 1D tensors, shard on the only dimension (dim 0)
                if tensor_rank == 1:
                    tensor_dim = 0
                
                placements.append(Shard(tensor_dim))
                needs_sharding = True
            except ValueError:
                placements.append(Replicate())
        else:
            placements.append(Replicate())
    
    # Ensure placements match mesh dimensions - pad with Replicate if needed
    if len(placements) < mesh_ndim:
        placements.extend([Replicate()] * (mesh_ndim - len(placements)))

    if not needs_sharding:
        # No sharding needed, replicate the tensor
        return torch.nn.Parameter(converted_tensor) if is_float_or_complex else converted_tensor

    # Use tensor parallel approach with ColwiseParallel/RowwiseParallel
    if TENSOR_PARALLEL_AVAILABLE and module_name is not None:
        # Determine the parallel style based on layout
        # For Linear layer weights (out_features, in_features):
        # - Keras (None, 'model') means shard output features -> ColwiseParallel
        # - Keras ('model', None) means shard input features -> RowwiseParallel
        
        parallel_style = _infer_parallel_style_from_layout(layout, tensor_rank)
        
        if parallel_style is not None:
            # Create DTensor with proper placement
            dtensor = torch_distribute_tensor(
                converted_tensor,
                device_mesh,
                placements
            )
            
            if not is_float_or_complex:
                return dtensor
            
            return torch.nn.Parameter(dtensor)

    # Fallback to regular DTensor distribution
    if DTENSOR_AVAILABLE and torch_distribute_tensor is not None:
        # Create DTensor-based Parameter
        dtensor = torch_distribute_tensor(
            converted_tensor,
            device_mesh,
            placements
        )

        # For non-floating point tensors, don't wrap in Parameter
        if not is_float_or_complex:
            return dtensor

        # Wrap as Parameter so it stays on device and tracks grads
        return torch.nn.Parameter(dtensor)
    else:
        # Manual sharding fallback (Slicing)
        return torch.nn.Parameter(converted_tensor)


def _infer_parallel_style_from_layout(layout, tensor_rank):
    """Infer the appropriate parallel style (ColwiseParallel/RowwiseParallel) from layout.
    
    This function translates Keras layout specifications to PyTorch tensor parallel styles.
    
    For a weight matrix:
    - Keras (None, 'model') -> shard output dimension -> ColwiseParallel
    - Keras ('model', None) -> shard input dimension -> RowwiseParallel
    
    Args:
        layout: Tuple of axis names (e.g., (None, 'model') or ('model',))
        tensor_rank: Rank of the tensor
    
    Returns:
        ColwiseParallel or RowwiseParallel or None
    """
    if not TENSOR_PARALLEL_AVAILABLE:
        return None
    
    # Find the position of 'model' axis in the layout
    model_idx = None
    for i, axis in enumerate(layout):
        if axis == 'model':
            model_idx = i
            break
    
    if model_idx is None:
        # No model axis specified, no tensor parallel needed
        return None
    
    # For weight matrices (typically 2D):
    # - Layout (None, 'model') with model_idx=1 means shard 2nd dim (output) -> ColwiseParallel
    # - Layout ('model', None) with model_idx=0 means shard 1st dim (input) -> RowwiseParallel
    
    if tensor_rank == 2:
        if model_idx == 1:
            # Keras: (None, 'model') -> shard output features
            # PyTorch Linear weight is (out_features, in_features)
            # ColwiseParallel shards the output dimension (first dim of weight)
            return ColwiseParallel()
        elif model_idx == 0:
            # Keras: ('model', None) -> shard input features
            # RowwiseParallel shards the input dimension (second dim of weight)
            return RowwiseParallel()
    elif tensor_rank == 1:
        # For bias vectors (1D), replicate
        return None
    
    return None


def _get_default_device_mesh() -> Optional[DeviceMesh]:
    """Get the default device mesh from global state."""
    mesh = global_state.get_global_attribute("torch_device_mesh", None)
    return mesh


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
    mesh_dim_names = device_mesh.mesh_dim_names
    return [
        Replicate() if axis is None else Shard(dim=mesh_dim_names.index(axis))
        for axis in axis_names
    ]


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
    return dtensor.redistribute(device_mesh, placements)


def _to_backend_mesh(device_mesh):
    """Converts a Keras DeviceMesh to a PyTorch DeviceMesh."""
    if not DTENSOR_AVAILABLE:
        return None

    # Create cache key from mesh configuration
    cache_key = f"torch_mesh_{device_mesh.shape}_{device_mesh.axis_names}"

    # Check cache first
    existing_mesh = global_state.get_global_attribute(cache_key)
    if existing_mesh is not None:
        global_state.set_global_attribute("torch_device_mesh", existing_mesh)
        return existing_mesh

    # Convert device strings to indices
    device_ids = [
        int(d.split(":")[-1]) if ":" in d else 0
        for d in device_mesh.devices.flatten()
    ]

    # Create backend mesh
    backend_mesh = TorchDeviceMesh(
        device_type="cuda" if torch.cuda.is_available() else "cpu",
        mesh=np.array(device_ids).reshape(device_mesh.shape),
        mesh_dim_names=device_mesh.axis_names
    )

    # Cache the mesh
    global_state.set_global_attribute(cache_key, backend_mesh)
    global_state.set_global_attribute("torch_device_mesh", backend_mesh)
    return backend_mesh


def _to_backend_layout(tensor_layout) -> tuple:
    """Convert the TensorLayout to PyTorch backend specific layout.
    
    Args:
        tensor_layout: TensorLayout instance to convert.
    
    Returns:
        Tuple of axis names for the layout.
    """
    if tensor_layout.device_mesh is None:
        raise ValueError(
            "Cannot create sharding when device mesh is not set "
            "for TensorLayout."
        )
    
    # Return the axes as-is (they will be used in distribute_tensor)
    return tensor_layout.axes


# Try to import tensor parallel functions
try:
    from torch.distributed.tensor.parallel import (
        parallelize_module,
        ColwiseParallel,
        RowwiseParallel,
        PrepareModuleInput,
        SequenceParallel,
    )
    TENSOR_PARALLEL_AVAILABLE = True
except ImportError:
    TENSOR_PARALLEL_AVAILABLE = False
    parallelize_module = None
    ColwiseParallel = None
    RowwiseParallel = None
    PrepareModuleInput = None
    SequenceParallel = None


def parallelize_torch_module(
    module: torch.nn.Module,
    device_mesh: DeviceMesh,
    layout_map: dict,
) -> torch.nn.Module:
    """Parallelize a PyTorch module using tensor parallelism.
    
    This is a thin wrapper around parallelize_module from PyTorch's
    tensor.parallel API.
    
    Args:
        module: A PyTorch nn.Module to parallelize
        device_mesh: A PyTorch DeviceMesh for distributed execution
        layout_map: A dict mapping parameter names to parallel styles.
            Example: {'weight': ColwiseParallel(), 'bias': RowwiseParallel()}
    
    Returns:
        A parallelized module that handles DTensor operations automatically.
    
    Raises:
        ImportError: If tensor parallel is not available
        ValueError: If device_mesh is not provided or is invalid
    """
    if not TENSOR_PARALLEL_AVAILABLE:
        raise ImportError(
            "PyTorch tensor.parallel is not available. "
            "Cannot use parallelize_torch_module. "
            "Please install PyTorch with tensor parallel support."
        )
    
    if device_mesh is None:
        raise ValueError("device_mesh cannot be None for parallelization")
    
    return parallelize_module(module, device_mesh, parallel_plan=layout_map)


def create_tp_plan_from_layout_map(
    module: torch.nn.Module,
    keras_layout_map: dict,
) -> dict:
    """Create a tensor parallel plan from a Keras-style layout map.
    
    This function translates Keras layout specifications (like 
    `(None, 'model')`) into PyTorch parallel styles 
    (`ColwiseParallel`, `RowwiseParallel`).
    
    Args:
        module: The PyTorch module being parallelized (to inspect layer types)
        keras_layout_map: Dict mapping parameter patterns to Keras sharding specs.
            Example: {'dense.*kernel': (None, 'model'), 'dense.*bias': ('model',)}
    
    Returns:
        A dict mapping parameter names to PyTorch parallel styles
    """
    if not keras_layout_map or not TENSOR_PARALLEL_AVAILABLE:
        return {}
    
    # Style mapping: model_idx -> parallel style
    styles = {0: RowwiseParallel(), 1: ColwiseParallel()}
    
    # Build plan using dict comprehension for path conversion
    plan = {}
    for pattern, sharding_spec in keras_layout_map.items():
        if sharding_spec is None:
            continue
        
        # Extract axes from TensorLayout if needed
        if hasattr(sharding_spec, 'axes'):
            sharding_spec = sharding_spec.axes
        
        # Convert path from Keras format (dense/kernel) to PyTorch (dense.weight)
        pytorch_pattern = pattern.replace('/', '.')
        
        if isinstance(sharding_spec, tuple):
            # Find position of 'model' axis
            model_idx = next((i for i, axis in enumerate(sharding_spec) if axis == 'model'), None)
            if model_idx is not None:
                plan[pytorch_pattern] = styles.get(model_idx, ColwiseParallel())
        elif isinstance(sharding_spec, str) and sharding_spec == 'model':
            plan[pytorch_pattern] = ColwiseParallel()
    
    return plan


def _to_dtensor(tensor, device_mesh=None, placements=None):
    """Convert a tensor to DTensor if it isn't already.

    This is a unified helper for converting regular tensors to DTensors.
    
    Args:
        tensor: torch.Tensor or DTensor to convert
        device_mesh: DeviceMesh to use for conversion. If None, uses default mesh.
        placements: Placements for the DTensor. If None, uses Replicate.

    Returns:
        DTensor (or original if already a DTensor or conversion not possible)
    """
    if not DTENSOR_AVAILABLE or tensor is None or isinstance(tensor, DTensor):
        return tensor

    if device_mesh is None:
        device_mesh = _get_default_device_mesh()

    if device_mesh is None:
        return tensor

    placements = [Replicate()] if placements is None else (placements if isinstance(placements, list) else [placements])
    return DTensor.from_local(tensor, device_mesh, placements)


def is_dtensor(tensor):
    """Check if a tensor is a DTensor.
    
    Args:
        tensor: torch.Tensor or DTensor to check
        
    Returns:
        bool: True if tensor is a DTensor, False otherwise
    """
    if not DTENSOR_AVAILABLE:
        return False
    return isinstance(tensor, DTensor)


def ensure_dtensor(tensor, device_mesh=None, placements=None):
    """Ensure a tensor is a DTensor, converting if necessary.
    
    Args:
        tensor: torch.Tensor or DTensor
        device_mesh: DeviceMesh to use for conversion. If None, uses default mesh.
        placements: Placements for the DTensor. If None, uses Replicate.
        
    Returns:
        DTensor (or original if already a DTensor or conversion not possible)
    """
    return _to_dtensor(tensor, device_mesh, placements)


def create_replicate_dtensor(tensor, device_mesh=None):
    """Create a replicated DTensor from a regular tensor.
    
    Args:
        tensor: torch.Tensor to convert
        device_mesh: DeviceMesh to use. If None, uses default mesh.
        
    Returns:
        Replicated DTensor (or original tensor if conversion not possible)
    """
    return _to_dtensor(tensor, device_mesh, [Replicate()])


def get_dtensor_local(tensor):
    """Get the local tensor from a DTensor.
    
    Args:
        tensor: DTensor or regular tensor
        
    Returns:
        Local tensor if input is DTensor, otherwise returns the input unchanged
    """
    if isinstance(tensor, DTensor):
        return tensor.to_local()
    return tensor


def dtensor_to_local(tensor):
    """Convert a DTensor to local tensor format.

    This function handles the conversion needed when using sharded DTensors.
    When a layer has DTensor weights, the output will be a DTensor that needs
    to be converted back to a local tensor for subsequent operations.

    Args:
        tensor: torch.Tensor, DTensor, or nested structure

    Returns:
        Same structure with DTensors converted to local tensors
    """
    if tensor is None:
        return tensor
    if isinstance(tensor, DTensor):
        return tensor.to_local()
    if isinstance(tensor, dict):
        return {k: dtensor_to_local(v) for k, v in tensor.items()}
    if isinstance(tensor, (list, tuple)):
        return type(tensor)(dtensor_to_local(v) for v in tensor)
    return tensor


def _should_auto_parallelize():
    """Check if automatic model parallelization should be performed.
    
    Returns:
        bool: True if auto-parallelization should happen
    """
    global _MODEL_PARALLELIZED
    
    if _MODEL_PARALLELIZED or not TENSOR_PARALLEL_AVAILABLE:
        return False
    
    from keras.src.distribution.distribution_lib import distribution, ModelParallel
    
    dist = distribution()
    if dist is None:
        return False
    
    # Skip for ModelParallel (distribute_variable handles it)
    if isinstance(dist, ModelParallel):
        return False
    
    # Check for layout map in non-ModelParallel distributions
    return hasattr(dist, '_layout_map') and dist._layout_map


def _auto_parallelize_model(model):
    """Automatically parallelize a model if conditions are met.
    
    This function is called during model build to automatically
    apply tensor parallelism when a ModelParallel distribution is active.
    
    Args:
        model: The Keras model to potentially parallelize
        
    Returns:
        The original model, possibly parallelized
    """
    if not _should_auto_parallelize():
        return model
    
    global _MODEL_PARALLELIZED
    try:
        parallelized = parallelize_keras_model(model)
        _MODEL_PARALLELIZED = True
        return parallelized
    except Exception:
        # Don't fail if auto-parallelization fails
        return model


def reset_model_parallelization_state():
    """Reset the model parallelization state.

    This can be used to force re-parallelization of a model.
    """
    global _MODEL_PARALLELIZED, _DISTRIBUTED_WEIGHTS_CREATED
    _MODEL_PARALLELIZED = _DISTRIBUTED_WEIGHTS_CREATED = False


def _get_tensor_parallel_mesh(device_mesh):
    """Extract a 1D DeviceMesh for tensor parallelism from a 2D mesh.
    
    PyTorch's parallelize_module requires a 1D DeviceMesh for tensor parallelism.
    Use device_mesh["tp"] for 2D meshes or return device_mesh if already 1D.
    
    Args:
        device_mesh: PyTorch DeviceMesh (can be 1D or 2D)
    
    Returns:
        A 1D DeviceMesh for tensor parallelism
    """
    if device_mesh is None:
        return None
    # If already 1D, return as-is
    if len(device_mesh.mesh_dim_names) == 1:
        return device_mesh
    # Use native slicing for 2D mesh (PyTorch 2.1+)
    return device_mesh["tp"]


def _get_keras_layout_map(layout_map):
    """Extract Keras layout map from various input formats."""
    if layout_map is None:
        from keras.src.distribution import distribution
        dist = distribution()
        if dist is not None and hasattr(dist, '_layout_map'):
            return dict(dist._layout_map)
        return {}
    if hasattr(layout_map, '_layout_map'):
        return dict(layout_map._layout_map)
    return dict(layout_map)


def _get_torch_module(model):
    """Extract the underlying PyTorch module from a Keras model."""
    return getattr(model, '_torch_layers', model)


def _parallelize_model(
    model: torch.nn.Module,
    device_mesh: DeviceMesh,
    layout_map: dict,
) -> torch.nn.Module:
    """Internal function to parallelize a model using tensor parallelism.
    
    Args:
        model: A Keras model or PyTorch module
        device_mesh: A PyTorch DeviceMesh for distributed execution
        layout_map: Dict mapping parameter patterns to Keras sharding specs
    
    Returns:
        A parallelized module
    """
    if not TENSOR_PARALLEL_AVAILABLE:
        raise ImportError(
            "PyTorch tensor.parallel is not available. "
            "Cannot use tensor parallelism. "
            "Please install PyTorch with tensor parallel support."
        )
    
    torch_module = _get_torch_module(model)
    keras_layout_map = _get_keras_layout_map(layout_map)
    
    # Create parallel plan from Keras layout map
    parallel_plan = create_tp_plan_from_layout_map(torch_module, keras_layout_map)
    if not parallel_plan:
        return model
    
    # Get 1D TP mesh (use device_mesh['tp'] or device_mesh if already 1D)
    tp_mesh = device_mesh["tp"] if len(device_mesh.mesh_dim_names) > 1 else device_mesh
    
    # Apply tensor parallelism
    return parallelize_module(torch_module, tp_mesh, parallel_plan=parallel_plan)


def parallelize_keras_model(
    model: torch.nn.Module,
    device_mesh=None,
    layout_map=None,
) -> torch.nn.Module:
    """Parallelize a Keras model using tensor parallelism.
    
    This function uses PyTorch's `torch.distributed.tensor.parallel.parallelize_module`
    to automatically handle DTensor conversions and weight sharding for the entire model.

    Args:
        model: A Keras model (must have a ._torch_layers attribute for PyTorch backend)
        device_mesh: A PyTorch DeviceMesh for distributed execution. If None, uses default.
        layout_map: A dict mapping parameter patterns to Keras sharding specs.
            If None, the layout_map from the distribution context is used.

    Returns:
        A parallelized module that handles DTensor operations automatically.

    Raises:
        ImportError: If tensor parallel is not available
        ValueError: If model or device_mesh is not provided
    """
    # Check for ModelParallel distribution (skip parallelize_module)
    from keras.src.distribution.distribution_lib import distribution, ModelParallel
    
    dist = distribution()
    is_model_parallel = isinstance(dist, ModelParallel)
    
    # Get device mesh
    if device_mesh is None:
        device_mesh = _get_default_device_mesh()
    
    if device_mesh is None:
        raise ValueError("device_mesh cannot be None for parallelization")
    
    # For ModelParallel, distribute_variable handles weight sharding
    if is_model_parallel:
        global _MODEL_PARALLELIZED
        _MODEL_PARALLELIZED = True
        return model
    
    # For non-ModelParallel, use tensor parallelism
    return _parallelize_model(model, device_mesh, layout_map)

