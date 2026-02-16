"""Distribution strategy utilities for PyTorch backend using DTensor.

This module provides distribution support for PyTorch backend using PyTorch's
DTensor API for distributed tensor computing.
"""

import os

import numpy as np
import torch

from keras.src.backend.common import global_state
from keras.src.backend.common.variables import standardize_dtype
from keras.src.backend.torch.core import convert_to_tensor, get_device, to_torch_dtype

from torch.distributed._tensor import DTensor, DeviceMesh, Replicate, Shard
from torch.distributed._tensor.api import distribute_tensor as torch_distribute_tensor
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel

TENSOR_PARALLEL_AVAILABLE = True

# Global variable to cache ModelParallel multi-process state
# This is set by TorchTrainer when entering fit/evaluate/predict
# and checked by prepare_input_for_distribution during training
_MP_MULTI_PROCESS_STATE = False


def set_mp_multi_process_state(is_mp_multi_process):
    """Set the cached ModelParallel multi-process state.
    
    This should be called by TorchTrainer at the start of fit/evaluate/predict
    when the distribution scope is still active.
    
    Args:
        is_mp_multi_process: Boolean indicating if we're in MP multi-process mode
    """
    global _MP_MULTI_PROCESS_STATE
    _MP_MULTI_PROCESS_STATE = is_mp_multi_process
    debug_mode = os.environ.get("KERAS_DISTRIBUTION_DEBUG", "0") == "1"
    if debug_mode:
        print(f"DEBUG | set_mp_multi_process_state: {_MP_MULTI_PROCESS_STATE}")


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
                # CRITICAL FIX: In distributed setting, each process can only see its local GPU
                # due to CUDA_VISIBLE_DEVICES isolation. We need to return ALL GPUs across
                # all ranks for proper mesh creation in DataParallel.
                # Use world_size to determine total GPU count when distributed is initialized.
                if torch.distributed.is_initialized():
                    world_size = torch.distributed.get_world_size()
                    # Each rank has 1 GPU due to CUDA_VISIBLE_DEVICES isolation
                    # Total GPUs = world_size * local_gpus_per_rank
                    local_gpu_count = torch.cuda.device_count()
                    total_gpus = world_size * local_gpu_count
                    # Return all GPU addresses across all ranks
                    devices.extend([f'cuda:{i}' for i in range(total_gpus)])
                else:
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
            # CRITICAL FIX: In distributed setting, each process can only see its local GPU
            # due to CUDA_VISIBLE_DEVICES isolation. We need to return total GPUs across
            # all ranks for proper mesh creation in DataParallel.
            if torch.distributed.is_initialized():
                world_size = torch.distributed.get_world_size()
                local_gpu_count = torch.cuda.device_count()
                return world_size * local_gpu_count
            return torch.cuda.device_count()

    if device_type in (None, 'cpu'):
        return os.cpu_count() or 1

    return 0


def initialize(job_addresses=None, num_processes=None, process_id=None):
    """Initialize the distribution system for multi-process settings.
    
    IMPORTANT: This function now properly initializes NCCL with appropriate
    timeout settings to prevent communication timeouts during DTensor operations.
    
    CRITICAL: This function sets CUDA device for the current rank BEFORE
    initializing distributed to prevent "Duplicate GPU detected" errors.
    
    V15 FIX: Use CUDA_VISIBLE_DEVICES to ensure unique GPU per process.
    This is the proper way to handle multi-process GPU assignment in PyTorch.
    """
    import datetime
    
    local_rank_env = os.environ.get("LOCAL_RANK")
    world_size_env = os.environ.get("WORLD_SIZE")

    if local_rank_env is not None and world_size_env is not None:
        num_processes = num_processes or int(world_size_env)
        process_id = process_id or int(local_rank_env)

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

    # CRITICAL: Set the CUDA device for this rank BEFORE any distributed init
    # This must be done before init_process_group to prevent NCCL errors
    
    # V15 FIX: Use proper GPU assignment via CUDA_VISIBLE_DEVICES
    # This is the recommended way by PyTorch documentation for multi-process GPU training
    if torch.cuda.is_available() and local_rank_env is not None:
        local_rank = int(local_rank_env)
        num_gpus = torch.cuda.device_count()
        
        # V15 FIX: Each process on the same machine MUST get a unique GPU
        # Use local_rank directly if local_rank < num_gpus (1:1 mapping)
        # If more processes than GPUs, use modulo (multiple processes per GPU)
        gpu_id = local_rank % num_gpus
        
        # V15 FIX: Set CUDA_VISIBLE_DEVICES to isolate this process to its GPU
        # This prevents NCCL from seeing duplicate GPUs
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        # Now set the current CUDA device
        torch.cuda.set_device(0)  # After CUDA_VISIBLE_DEVICES, only GPU 0 is visible
        
        print(f"[initialize() Rank {local_rank}] Set CUDA device to GPU {gpu_id} (visible as 0), num_gpus={num_gpus}")

    # Set NCCL environment variables for better performance and reliability
    # V15 FIX: Reduce NCCL debug verbosity to avoid clutter
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
    
    # Set NCCL timeout to prevent hangs during initialization
    # Default is 10 minutes, we set it to 30 minutes for safety
    os.environ.setdefault("NCCL_TIMEOUT", "1800")

    init_method = f"tcp://{job_addresses}" if job_addresses else "env://"
    if "," in str(job_addresses):
        init_method = f"tcp://{job_addresses.split(',')[0]}"

    # Initialize NCCL with proper timeout (30 minutes)
    timeout = datetime.timedelta(seconds=1800)
    torch.distributed.init_process_group(
        backend="nccl" if torch.cuda.is_available() else "gloo",
        init_method=init_method,
        world_size=num_processes,
        rank=process_id or 0,
        timeout=timeout,
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

    # Handle TensorLayout objects by extracting the axes tuple
    from keras.src.distribution.distribution_lib import TensorLayout
    if isinstance(backend_layout, TensorLayout):
        # Convert TensorLayout to placements using _layout_to_placements
        placements = _layout_to_placements(backend_layout, tensor, device_mesh)
    elif isinstance(backend_layout, tuple):
        placements = _axis_names_to_placements(backend_layout, device_mesh) if backend_layout else None
    else:
        placements = None
    
    if placements:
        return tensor.redistribute(device_mesh, placements) if isinstance(tensor, DTensor) else torch_distribute_tensor(tensor, device_mesh, placements)

    return tensor


def distribute_variable(tensor, layout=None):
    """Distributes a Keras variable using PyTorch DTensor."""
    from keras.src.distribution.distribution_lib import distribution

    converted_tensor = convert_to_tensor(tensor)
    
    # Check if tensor is already a DTensor and get local tensor for dtype check
    from keras.src.backend.torch.distribution_lib import is_dtensor, get_dtensor_local
    local_tensor = None
    if is_dtensor(converted_tensor):
        local_tensor = get_dtensor_local(converted_tensor)
        # Use local tensor's dtype for float/complex check since that's what
        # PyTorch actually uses when creating Parameters
        is_float_or_complex = local_tensor.dtype.is_floating_point or local_tensor.dtype.is_complex
    else:
        is_float_or_complex = converted_tensor.dtype.is_floating_point or converted_tensor.dtype.is_complex

    # Check if distribution is disabled
    if os.environ.get("KERAS_DISTRIBUTION_DISABLE", "0") == "1":
        return torch.nn.Parameter(converted_tensor) if is_float_or_complex else converted_tensor

    # Debug logging - define at function level
    debug_mode = os.environ.get("KERAS_DISTRIBUTION_DEBUG", "0") == "1"
    rank = 0
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
    except:
        pass

    if debug_mode:
        print(f"DEBUG | [Rank {rank}] distribute_variable() called")
        print(f"DEBUG | [Rank {rank}]   - tensor shape: {converted_tensor.shape}")
        print(f"DEBUG | [Rank {rank}]   - tensor type: {type(converted_tensor)}")
        if isinstance(converted_tensor, DTensor):
            print(f"DEBUG | [Rank {rank}]   - existing placements: {converted_tensor.placements}")
        print(f"DEBUG | [Rank {rank}]   - layout: {layout}")
        print(f"DEBUG | [Rank {rank}]   - is_float_or_complex: {is_float_or_complex}")

    current_distribution = distribution()
    if current_distribution is not None:
        device_mesh = _to_backend_mesh(current_distribution.device_mesh)
        if device_mesh is not None:
            # CRITICAL FIX: Handle TensorLayout objects properly
            from keras.src.distribution.distribution_lib import TensorLayout
            if isinstance(layout, TensorLayout):
                # Use the axes from TensorLayout
                layout_axes = layout.axes
            else:
                layout_axes = layout
            
            placements = _layout_to_placements(layout_axes, converted_tensor, device_mesh) if layout_axes else None
            
            if debug_mode:
                print(f"DEBUG | [Rank {rank}]   - device_mesh: {device_mesh}")
                print(f"DEBUG | [Rank {rank}]   - target placements: {placements}")
            
            # Handle case where tensor is already a DTensor
            if isinstance(converted_tensor, DTensor):
                if placements:
                    # Check if redistribution is needed
                    current_placements = converted_tensor.placements
                    needs_redistribute = False
                    for i, (curr, target) in enumerate(zip(current_placements, placements)):
                        # Check if current and target are different types
                        curr_is_replicate = isinstance(curr, Replicate)
                        curr_is_shard = isinstance(curr, Shard)
                        target_is_replicate = isinstance(target, Replicate)
                        target_is_shard = isinstance(target, Shard)
                        
                        if (curr_is_replicate and target_is_shard) or (curr_is_shard and target_is_replicate):
                            needs_redistribute = True
                            break
                        elif curr_is_shard and target_is_shard:
                            # Check if shard dimension is different
                            if curr.dim != target.dim:
                                needs_redistribute = True
                                break
                    
                    if needs_redistribute:
                        if debug_mode:
                            print(f"DEBUG | [Rank {rank}]   - Redistributing DTensor from {current_placements} to {placements}")
                        try:
                            dtensor = converted_tensor.redistribute(device_mesh, placements)
                        except Exception as e:
                            if debug_mode:
                                print(f"DEBUG | [Rank {rank}]   - Redistribute failed: {e}, using local tensor")
                            # Fallback: convert from local and redistribute
                            local_tensor = converted_tensor.to_local()
                            dtensor = torch_distribute_tensor(local_tensor, device_mesh, placements)
                    else:
                        if debug_mode:
                            print(f"DEBUG | [Rank {rank}]   - No redistribution needed, placements match")
                        dtensor = converted_tensor
                else:
                    if debug_mode:
                        print(f"DEBUG | [Rank {rank}]   - No target placements, returning as-is")
                    dtensor = converted_tensor
                
                if debug_mode:
                    print(f"DEBUG | [Rank {rank}]   - Final DTensor local shape: {dtensor.to_local().shape}")
                return torch.nn.Parameter(dtensor) if is_float_or_complex else dtensor
            
            # Handle case where tensor is not yet a DTensor
            if placements and any(isinstance(p, Shard) for p in placements):
                dtensor = torch_distribute_tensor(converted_tensor, device_mesh, placements)
                if debug_mode:
                    print(f"DEBUG | [Rank {rank}]   - Created DTensor with local shape: {dtensor.to_local().shape}")
                return torch.nn.Parameter(dtensor) if is_float_or_complex else dtensor
            elif debug_mode:
                print(f"DEBUG | [Rank {rank}]   - No sharding placements, returning as-is or replicated")

    if debug_mode:
        print(f"DEBUG | [Rank {rank}]   - No distribution active, returning regular tensor")
    return torch.nn.Parameter(converted_tensor) if is_float_or_complex else converted_tensor


def _layout_to_placements(layout, tensor, device_mesh):
    """Convert Keras layout tuple to DTensor placements.
    
    IMPORTANT: PyTorch DTensor requires placements to have the same length
    as device_mesh.ndim, NOT the layout length or tensor rank.
    
    For model parallelism with 1D mesh (shape=(2,)):
    - layout (None, 'model') should produce ONE placement: [Shard(1)]
      This means shard on TENSOR dimension 1, using the single mesh dimension
    - layout () should produce ONE placement: [Replicate()]
    
    The 'model' axis in the layout specifies which TENSOR dimension should be
    sharded across the model-parallel devices.
    
    Example:
    - For query kernel with shape (768, 12, 64) and layout (None, 'model'):
      - We want to shard on the 2nd dimension (12 -> 6+6)
      - Layout position 1 has 'model', so we return [Shard(1)]
      - This shards tensor dimension 1 across the 2 model-parallel devices
    """
    tensor_rank = tensor.dim()
    mesh_ndim = device_mesh.mesh.ndim
    
    # CRITICAL FIX: Handle TensorLayout objects properly
    # The layout can be a TensorLayout object, tuple, or None
    from keras.src.distribution.distribution_lib import TensorLayout
    if isinstance(layout, TensorLayout):
        # Get the axes tuple from the TensorLayout object
        layout_axes = layout.axes
    else:
        layout_axes = layout
    
    # For 1D mesh, return exactly 1 placement
    if mesh_ndim == 1:
        # Look for 'model' axis in the layout
        if layout_axes is not None:
            for i, axis in enumerate(layout_axes):
                if axis == 'model':
                    # Found 'model' axis at layout position i
                    # Shard on TENSOR dimension i (not mesh dimension)
                    return [Shard(i)]
        # No 'model' axis found or layout is None - replicate
        return [Replicate()]
    else:
        # Multi-dimensional mesh case
        # Return exactly mesh_ndim placements
        placements = []
        for i in range(mesh_ndim):
            if layout_axes is not None and i < len(layout_axes):
                axis = layout_axes[i]
                if axis is None:
                    placements.append(Replicate())
                elif axis == 'model':
                    placements.append(Shard(i))
                else:
                    placements.append(Replicate())
            else:
                placements.append(Replicate())
        return placements


def _get_default_device_mesh():
    """Get the default device mesh from global state.
    
    IMPORTANT: This function must use the same cache key logic as _to_backend_mesh()
    to ensure we retrieve the correct mesh for the current distribution type
    (DataParallel vs ModelParallel).
    
    CRITICAL FIX: We must verify the mesh matches the current distribution type.
    If the current distribution is ModelParallel but we have a cached DataParallel mesh
    (or vice versa), we should not use that cached mesh as it will cause cross-mesh
    operations which DTensor does not support.
    """
    from keras.src.distribution.distribution_lib import distribution, DataParallel, ModelParallel
    
    # Build the same cache key as _to_backend_mesh() to ensure we get the right mesh
    # for the current distribution type
    current_dist = distribution()
    
    # CRITICAL FIX: First check if current distribution is active and has a mesh
    if current_dist is not None and hasattr(current_dist, 'device_mesh'):
        device_mesh = current_dist.device_mesh
        
        # Build the same cache key as _to_backend_mesh()
        dist_type = ""
        if isinstance(current_dist, ModelParallel):
            dist_type = "MP"
        elif isinstance(current_dist, DataParallel):
            dist_type = "DP"
        else:
            dist_type = "NONE"
        
        cache_key = f"torch_mesh_{device_mesh.shape}_{device_mesh.axis_names}_{dist_type}"
        cached = global_state.get_global_attribute(cache_key)
        if cached is not None:
            return cached

        return None  # Signal that we need to convert

    generic_cached = global_state.get_global_attribute("torch_device_mesh", None)
    if generic_cached is not None and current_dist is not None:
        if isinstance(current_dist, ModelParallel):
            if hasattr(generic_cached, 'mesh') and generic_cached.mesh.ndim == 1:
                return None
        elif isinstance(current_dist, DataParallel):
            if hasattr(generic_cached, 'mesh') and generic_cached.mesh.ndim > 1:
                return None
    
    return generic_cached


def _axis_names_to_placements(axis_names, device_mesh):
    """Convert axis names to DTensor placements.
    
    IMPORTANT: PyTorch DTensor requires placements to have the same length
    as device_mesh.ndim, NOT the length of axis_names.
    
    For a 1D mesh (mesh.ndim == 1), we return only ONE placement:
    - If 'model' axis is in axis_names, return [Shard(0)] to shard on that mesh dim
    - Otherwise return [Replicate()]
    
    For multi-dimensional meshes, we return mesh_ndim placements.
    """
    mesh_ndim = device_mesh.mesh.ndim
    mesh_dim_names = device_mesh.mesh_dim_names
    
    # For 1D mesh, return exactly 1 placement
    if mesh_ndim == 1:
        # Check if the single mesh dimension ('model') is in the axis names
        if 'model' in axis_names:
            return [Shard(0)]
        return [Replicate()]
    else:
        # Multi-dimensional mesh case
        # Return exactly mesh_ndim placements
        placements = []
        for i in range(mesh_ndim):
            if i < len(axis_names):
                axis = axis_names[i]
                if axis is None:
                    placements.append(Replicate())
                elif axis in mesh_dim_names:
                    placements.append(Shard(mesh_dim_names.index(axis)))
                else:
                    placements.append(Replicate())
            else:
                placements.append(Replicate())
        return placements


def _to_backend_mesh(device_mesh):
    """Converts a Keras DeviceMesh to a PyTorch DeviceMesh.
    
    IMPORTANT: For multi-process setups, each process can only see its local GPU.
    We use init_device_mesh which properly handles this case.
    
    For ModelParallel with multi-dimensional meshes, we need to handle the mesh
    differently to support tensor parallelism across multiple GPUs.
    """
    from torch.distributed.device_mesh import init_device_mesh
    
    # Get the current distribution to ensure we're caching the right mesh
    from keras.src.distribution.distribution_lib import distribution, DataParallel, ModelParallel
    current_dist = distribution()
    
    # Debug logging
    debug_mode = os.environ.get("KERAS_DISTRIBUTION_DEBUG", "0") == "1"
    if debug_mode:
        rank = 0
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
        except:
            pass
        print(f"DEBUG | [Rank {rank}] _to_backend_mesh: device_mesh={device_mesh}, current_dist={current_dist}")
    
    # Build a more specific cache key that includes the distribution type
    # This ensures different distributions get different cache entries
    dist_type = ""
    if isinstance(current_dist, ModelParallel):
        dist_type = "MP"
    elif isinstance(current_dist, DataParallel):
        dist_type = "DP"
    else:
        dist_type = "NONE"
    
    cache_key = f"torch_mesh_{device_mesh.shape}_{device_mesh.axis_names}_{dist_type}"
    cached = global_state.get_global_attribute(cache_key)
    if cached is not None:
        if debug_mode:
            print(f"DEBUG | [Rank {rank}] _to_backend_mesh: using cached mesh: {cached}")
        global_state.set_global_attribute("torch_device_mesh", cached)
        return cached

    # Get local rank for proper CUDA device mapping
    local_rank = 0
    if torch.distributed.is_initialized():
        local_rank = torch.distributed.get_rank()
        if torch.cuda.is_available():
            # Ensure we use the correct CUDA device for this process
            torch.cuda.set_device(local_rank)

    # For multi-process setups, use init_device_mesh
    if torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        if world_size > 1:
            # Each process has its own local GPU
            # CRITICAL FIX: Determine the correct mesh dimension name based on
            # the distribution type (DataParallel vs ModelParallel)
            # For DataParallel: use "batch" as the axis name
            # For ModelParallel: use the full mesh structure (can be multi-dimensional)
            from keras.src.distribution.distribution_lib import distribution, DataParallel, ModelParallel
            current_dist = distribution()
            
            # Check if this is ModelParallel with a multi-dimensional mesh
            is_model_parallel = isinstance(current_dist, ModelParallel)
            is_multi_dim_mesh = len(device_mesh.axis_names) > 1
            
            if debug_mode:
                print(f"DEBUG | [Rank {rank}] _to_backend_mesh: is_model_parallel={is_model_parallel}, is_multi_dim_mesh={is_multi_dim_mesh}, world_size={world_size}")
            
            if is_model_parallel and is_multi_dim_mesh:
                # For ModelParallel with multi-dimensional mesh in multi-process mode,
                # we need to fall back to 1D mesh because each process can only see
                # its local GPU due to CUDA_VISIBLE_DEVICES isolation.
                #
                # The 2D mesh (batch, model) requires multiple GPUs visible per process,
                # but in multi-process mode each process only has 1 GPU.
                #
                # Instead, we create a 1D mesh and use the placement logic to handle
                # the sharding. This ensures consistency between inputs and model weights.
                
                # Get the mesh dimension names - use "model" for MP
                mesh_dim_names = ["model"]
                
                if debug_mode:
                    print(f"DEBUG | [Rank {rank}] _to_backend_mesh: creating 1D mesh for MP in multi-process mode (fallback from 2D)")
                
                if torch.cuda.is_available():
                    # Create 1D mesh with world_size (each process has 1 GPU)
                    backend_mesh = init_device_mesh(
                        device_type="cuda",
                        mesh_shape=(world_size,),
                        mesh_dim_names=mesh_dim_names
                    )
                    
                    if debug_mode:
                        print(f"DEBUG | [Rank {rank}] _to_backend_mesh: created 1D mesh for MP: {backend_mesh}")
                    
                    global_state.set_global_attribute(cache_key, backend_mesh)
                    global_state.set_global_attribute("torch_device_mesh", backend_mesh)
                    return backend_mesh
            else:
                # DataParallel or single-dimensional mesh
                # Use 1D mesh where each process has one device
                if isinstance(current_dist, DataParallel):
                    # DataParallel uses "batch" axis for data parallelism
                    mesh_dim_names = [current_dist.batch_dim_name]
                else:
                    # ModelParallel with 1D mesh or other distributions use "model"
                    mesh_dim_names = ["model"]
                
                if debug_mode:
                    print(f"DEBUG | [Rank {rank}] _to_backend_mesh: creating 1D mesh with dim_names={mesh_dim_names}")
                
                if torch.cuda.is_available():
                    backend_mesh = init_device_mesh(
                        device_type="cuda",
                        mesh_shape=(world_size,),
                        mesh_dim_names=mesh_dim_names
                    )
                    global_state.set_global_attribute(cache_key, backend_mesh)
                    global_state.set_global_attribute("torch_device_mesh", backend_mesh)
                    return backend_mesh
    
    # For single-process, create DeviceMesh with all GPUs
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    device_ids = [int(d.split(":")[-1]) if ":" in d else 0 for d in device_mesh.devices.flatten()]
    mesh_tensor = torch.tensor(device_ids, dtype=torch.int64).reshape(device_mesh.shape)
    
    backend_mesh = DeviceMesh(
        device_type=device_type,
        mesh=mesh_tensor,
        mesh_dim_names=device_mesh.axis_names
    )
    
    if debug_mode:
        print(f"DEBUG | [Rank {rank}] _to_backend_mesh: created single-process mesh: {backend_mesh}")

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
        if to_dtensor:
            # CRITICAL FIX: Get the device mesh from the current distribution's scope,
            # not from global cache. The issue is that after DataParallel scope exits,
            # the global cache might still have the old mesh. We need to use the mesh
            # from the currently active distribution (if any), not the last cached one.
            from keras.src.distribution.distribution_lib import distribution, ModelParallel, DataParallel
            
            # Get the current distribution - this checks the scope, not global cache
            current_dist = distribution()
            
            # CRITICAL FIX: For ModelParallel with multi-dimensional mesh, we need to
            # use the same mesh as the model weights. The _to_backend_mesh function
            # should return the correct 2D mesh for ModelParallel.
            torch_device_mesh = None
            
            if current_dist is not None and hasattr(current_dist, 'device_mesh'):
                # Use the current distribution's device mesh - this is the correct approach
                # because we want inputs to match the model weights' mesh
                torch_device_mesh = _to_backend_mesh(current_dist.device_mesh)
                
                # Debug: Log which mesh we're using
                debug_mode = os.environ.get("KERAS_DISTRIBUTION_DEBUG", "0") == "1"
                if debug_mode:
                    rank = 0
                    try:
                        import torch.distributed as dist
                        if dist.is_available() and dist.is_initialized():
                            rank = dist.get_rank()
                    except:
                        pass
                    print(f"DEBUG | [Rank {rank}] _convert_structure: using current_dist mesh: {torch_device_mesh}")
            elif device_mesh is not None:
                # Fallback for non-distributed cases or when no distribution is active
                # Check mesh dimensionality
                if hasattr(device_mesh, 'mesh'):
                    # This is already a PyTorch DeviceMesh
                    torch_device_mesh = device_mesh
                else:
                    # This is a Keras DeviceMesh
                    torch_device_mesh = _to_backend_mesh(device_mesh)
                
                # CRITICAL FIX: Don't use global cache if current_dist is ModelParallel
                # The global cache might have a stale DataParallel mesh
                if current_dist is not None and isinstance(current_dist, ModelParallel):
                    # We're in ModelParallel but couldn't get the right mesh from current_dist
                    # This shouldn't happen, but let's be safe and not use wrong mesh
                    debug_mode = os.environ.get("KERAS_DISTRIBUTION_DEBUG", "0") == "1"
                    if debug_mode:
                        print(f"DEBUG | [Rank 0] _convert_structure: WARNING - current_dist is MP but mesh came from fallback")
                    # Return as-is to avoid cross-mesh operations
                    return x
            
            if torch_device_mesh is not None:
                mesh_ndim = 1
                if hasattr(torch_device_mesh, 'mesh'):
                    mesh_ndim = torch_device_mesh.mesh.ndim
                
                # CRITICAL FIX: For multi-dimensional mesh (ModelParallel), we need
                # to use the correct number of placements that matches the mesh ndim.
                # This ensures inputs use the same mesh as model weights.
                # 
                # For multi-process with 2D mesh, inputs should be replicated across
                # both dimensions (batch and model) to match the full model parallelism.
                if mesh_ndim == 1:
                    placements = [Replicate()]
                else:
                    # For 2D mesh in multi-process mode, replicate on all dimensions
                    # This ensures the input is available on all devices for model parallelism
                    placements = [Replicate()] * mesh_ndim
                
                # CRITICAL FIX: Ensure the local tensor is on the correct device
                # In multi-process mode with CUDA_VISIBLE_DEVICES, each process
                # only sees its local GPU as cuda:0. We need to ensure the local
                # tensor is on the correct device before creating the DTensor.
                local_tensor = x
                
                # Handle both numpy arrays and torch tensors - ensure they're on the right device
                if isinstance(x, np.ndarray):
                    # Convert numpy array to torch tensor first, then handle device
                    if x.dtype == np.uint32:
                        x = x.astype(np.int64)
                    if standardize_dtype(x.dtype) == "bfloat16":
                        x = x.astype(np.float32)
                        dtype = "bfloat16"
                    dtype = dtype or x.dtype
                    
                    # Get the correct local device for this process
                    if torch.distributed.is_initialized() and torch.cuda.is_available():
                        local_device = f"cuda:{torch.cuda.current_device()}"
                    else:
                        local_device = get_device()
                    
                    local_tensor = torch.as_tensor(x, dtype=to_torch_dtype(dtype), device=local_device)
                elif isinstance(x, torch.Tensor):
                    # For existing torch tensors, ensure they're on the correct device
                    if torch.distributed.is_initialized() and torch.cuda.is_available():
                        local_device = torch.cuda.current_device()
                        if x.is_cuda:
                            if x.device.index != local_device:
                                local_tensor = x.to(f"cuda:{local_device}")
                        else:
                            # Tensor is on CPU or different device, move to correct CUDA device
                            local_tensor = x.to(f"cuda:{local_device}")
                
                return DTensor.from_local(local_tensor, torch_device_mesh, placements)
            
            # If no mesh available, return as-is
            return x

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
    """Convert inputs to DTensors when model has DTensor weights.
    
    For model parallelism with sharded weights, inputs need to be handled carefully.
    In multi-process mode, each process should keep its inputs local rather than
    converting to DTensors, since the model weights are sharded and each process
    operates on its local portion.
    
    IMPORTANT: For ModelParallel (tensor parallelism), each rank should receive
    the FULL batch, not a slice. Only the model weights are sharded across
    the "model" axis. This is different from DataParallel where the batch IS
    split across ranks.
    
    This function checks if we have an active device mesh and distributed
    context, and converts inputs to DTensors accordingly.
    """
    from keras.src.distribution.distribution_lib import distribution, ModelParallel, DataParallel
    
    # Get the current distribution from the scope context, not from global cache
    current_dist = distribution()
    
    # Debug logging - check what distribution we have
    debug_mode = os.environ.get("KERAS_DISTRIBUTION_DEBUG", "0") == "1"
    if debug_mode:
        rank = 0
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
        except:
            pass
        print(f"DEBUG | [Rank {rank}] prepare_input_for_distribution: current_dist={current_dist}, type={type(current_dist)}")
    
    # Check if we have a ModelParallel distribution active
    is_mp = isinstance(current_dist, ModelParallel)
    
    if debug_mode:
        print(f"DEBUG | [Rank {rank}] prepare_input_for_distribution: is_mp={is_mp}")
    
    # Also check if torch distributed is initialized
    # Even outside the scope, we might have sharded weights
    is_distributed = torch.distributed.is_initialized()
    
    # CRITICAL FIX: Check the cached ModelParallel multi-process state.
    # This is set by TorchTrainer at the start of fit/evaluate/predict when
    # the distribution scope is still active, and is used when the scope
    # is not available (e.g., during torch.compile traced execution).
    global _MP_MULTI_PROCESS_STATE
    cached_mp_multi_process = _MP_MULTI_PROCESS_STATE
    
    if debug_mode:
        print(f"DEBUG | [Rank {rank}] prepare_input_for_distribution: cached_mp_multi_process={cached_mp_multi_process}")
    
    # CRITICAL FIX: For ModelParallel in multi-process mode, do NOT convert inputs
    # to DTensors. Each process should keep its inputs local. The model weights
    # are sharded across devices, and each process only needs its local input
    # to compute with its portion of the sharded weights.
    #
    # The key is that inputs should NOT be distributed - they should stay as local
    # tensors on each rank. The Dense layer will automatically handle the DTensor
    # weights because PyTorch's DTensor matmul can broadcast a local input with
    # sharded weights.
    #
    # Check both the active distribution AND the cached state
    if (is_mp or cached_mp_multi_process) and is_distributed:
        if debug_mode:
            rank = 0
            try:
                import torch.distributed as dist
                if dist.is_available() and dist.is_initialized():
                    rank = dist.get_rank()
            except:
                pass
            print(f"DEBUG | [Rank {rank}] prepare_input_for_distribution: skipping DTensor conversion for ModelParallel in multi-process mode (is_mp={is_mp}, cached_mp_multi_process={cached_mp_multi_process})")
        
        # CRITICAL FIX: Ensure the input tensor is on the correct CUDA device
        # In multi-process mode with CUDA_VISIBLE_DEVICES, each process
        # only sees its local GPU as cuda:0. We need to ensure the local
        # tensor is on the correct device.
        if isinstance(x, torch.Tensor):
            local_rank = 0
            try:
                import torch.distributed as dist
                if dist.is_available() and dist.is_initialized():
                    local_rank = dist.get_rank()
            except:
                pass
            
            if torch.cuda.is_available():
                # Ensure tensor is on the correct CUDA device for this process
                if x.is_cuda:
                    current_device = torch.cuda.current_device()
                    if x.device.index != current_device:
                        x = x.to(f"cuda:{current_device}")
                else:
                    # Tensor is on CPU, move to CUDA
                    x = x.to(device=f"cuda:{local_rank}")
        elif isinstance(x, np.ndarray):
            # Convert numpy array to torch tensor on correct device
            if torch.cuda.is_available():
                local_rank = 0
                try:
                    import torch.distributed as dist
                    if dist.is_available() and dist.is_initialized():
                        local_rank = dist.get_rank()
                except:
                    pass
                
                # Convert numpy to torch tensor on correct device
                if x.dtype == np.uint32:
                    x = x.astype(np.int64)
                dtype = standardize_dtype(x.dtype)
                if dtype == "bfloat16":
                    x = x.astype(np.float32)
                    dtype = "bfloat16"
                from keras.src.backend.torch.core import to_torch_dtype
                x = torch.as_tensor(x, dtype=to_torch_dtype(dtype), device=f"cuda:{local_rank}")
        
        return x
    
    # CRITICAL FIX: Get the device mesh from the CURRENT distribution, not from
    # global cache. This ensures inputs use the same mesh as the model weights.
    # We need to use _to_backend_mesh to convert the Keras DeviceMesh to PyTorch DeviceMesh
    # with the correct cache key for ModelParallel.
    if current_dist is not None and hasattr(current_dist, 'device_mesh'):
        # Use _to_backend_mesh to get the correct 2D mesh for ModelParallel
        torch_device_mesh = _to_backend_mesh(current_dist.device_mesh)
        
        if debug_mode:
            print(f"DEBUG | [Rank {rank}] prepare_input_for_distribution: got torch_device_mesh from current_dist: {torch_device_mesh}")
    else:
        # Fallback to global cache
        torch_device_mesh = _get_default_device_mesh()
        
        if debug_mode:
            print(f"DEBUG | [Rank {rank}] prepare_input_for_distribution: got torch_device_mesh from fallback: {torch_device_mesh}")
    
    # Debug logging
    if debug_mode and torch_device_mesh is not None:
        rank = 0
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
        except:
            pass
        print(f"DEBUG | [Rank {rank}] prepare_input_for_distribution: FINAL torch_device_mesh={torch_device_mesh}, is_mp={is_mp}")
    
    # Convert to DTensor if:
    # 1. We have a device mesh AND
    # 2. Either ModelParallel is active OR distributed is initialized
    # Note: For ModelParallel in multi-process mode, we skip this as handled above
    if torch_device_mesh is not None and (is_mp or is_distributed):
        return _convert_structure(x, torch_device_mesh, to_dtensor=True, gather_sharded=False)
    
    return x


def prepare_output_for_loss(x):
    """Convert DTensor outputs to local tensors.
    
    For ModelParallel training, we need to handle sharded outputs properly.
    When the distribution scope is not active but we have distributed initialized,
    we still need to handle DTensor outputs correctly.
    """
    from keras.src.distribution.distribution_lib import distribution, ModelParallel
    
    # Check if we have an active ModelParallel distribution OR if we had one
    # (by checking if the mesh was cached)
    current_dist = distribution()
    is_mp = isinstance(current_dist, ModelParallel)
    
    # Even if the scope has exited, check if we have a cached ModelParallel mesh
    if not is_mp and torch.distributed.is_initialized():
        # Check if there's a cached MP mesh
        cached_mesh = global_state.get_global_attribute("torch_device_mesh", None)
        if cached_mesh is not None and hasattr(cached_mesh, 'mesh'):
            # Check if it's a 1D mesh (which is what MP uses in multi-process)
            if cached_mesh.mesh.ndim == 1:
                # This is likely a cached MP mesh, treat as MP
                is_mp = True
    
    if not is_mp:
        # Not ModelParallel, return as-is
        if isinstance(x, DTensor):
            return x.to_local()
        return x
    
    # For ModelParallel, convert sharded DTensors to local tensors
    return _convert_structure(x, None, to_dtensor=False, gather_sharded=True)

