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
            if placements:
                # CRITICAL FIX: Always create DTensor when placements is specified,
                # regardless of whether it's Sharded or Replicated.
                # This ensures all variables (even replicated ones) become DTensors,
                # which is necessary for compatibility with sharded variables.
                # For example, when token_embedding (sharded DTensor) is added to
                # position_embedding (replicated DTensor), both must be DTensors.
                dtensor = torch_distribute_tensor(converted_tensor, device_mesh, placements)
                if debug_mode:
                    print(f"DEBUG | [Rank {rank}]   - Created DTensor with placements: {placements}")
                return torch.nn.Parameter(dtensor) if is_float_or_complex else dtensor
            elif debug_mode:
                print(f"DEBUG | [Rank {rank}]   - No placements, returning as-is")

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
                    # Use the layout position 'i' as the tensor dimension to shard.
                    # If the layout position is larger than the tensor rank, fall back
                    # to sharding the last tensor dimension to avoid invalid dim errors.
                    shard_dim = i if i < tensor.dim() else max(0, tensor.dim() - 1)
                    return [Shard(shard_dim)]
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
    
    CRITICAL FIX 2: Even when the distribution scope is not active (e.g., during
    model forward pass when computing causal masks), we should still detect if
    ModelParallel is active by checking the _MP_MULTI_PROCESS_STATE global flag.
    
    CRITICAL FIX 3: This function should ALSO return the cached mesh when
    distributed is initialized but current_dist is None. This handles the case
    where distribute_tensor is called during model forward pass AFTER the 
    distribution scope has exited but we still have a valid cached mesh.
    """
    from keras.src.distribution.distribution_lib import distribution, DataParallel, ModelParallel
    
    # Build the same cache key as _to_backend_mesh() to ensure we get the right mesh
    # for the current distribution type
    current_dist = distribution()
    
    # Check if we have a cached ModelParallel multi-process state
    # This is set by TorchTrainer at the start of fit/evaluate/predict when
    # the distribution scope is still active, and is used when the scope
    # is not available (e.g., during torch.compile traced execution or
    # when computing causal masks inside the model).
    global _MP_MULTI_PROCESS_STATE
    is_mp_multi_process = _MP_MULTI_PROCESS_STATE
    
    # CRITICAL FIX: Also check if torch distributed is initialized
    is_distributed = torch.distributed.is_initialized()
    
    # CRITICAL FIX: Check for cached mesh FIRST.
    # This handles the case where we're in multi-process mode but the distribution
    # scope has exited (e.g., during model forward pass). We should return the
    # cached mesh if it exists.
    if is_distributed:
        cached_mesh = global_state.get_global_attribute("torch_device_mesh", None)
        if cached_mesh is not None and hasattr(cached_mesh, 'mesh'):
            # Check if we are in meta scope
            if get_device() == "meta" and cached_mesh.device_type != "meta":
                # Create a meta version of the mesh for symbolic building
                # This ensures consistent device types between mesh and tensors
                meta_cached = global_state.get_global_attribute("torch_device_mesh_meta", None)
                if meta_cached is not None:
                    return meta_cached
                meta_mesh = DeviceMesh("meta", cached_mesh.mesh, mesh_dim_names=cached_mesh.mesh_dim_names)
                global_state.set_global_attribute("torch_device_mesh_meta", meta_mesh)
                return meta_mesh
                
            # If we're in multi-process MP mode (cached flag), we need a 1D mesh.
            if is_mp_multi_process and cached_mesh.mesh.ndim == 1:
                return cached_mesh
            # If we have any valid mesh and we're distributed, it's better than None
            # because regular tensors cause mixed-tensor errors.
            return cached_mesh
    
    # Try to detect ModelParallel from current distribution scope
    if current_dist is not None and hasattr(current_dist, 'device_mesh'):
        device_mesh = current_dist.device_mesh
        
        # Build the same cache key as _to_backend_mesh()
        dist_type = "MP" if isinstance(current_dist, ModelParallel) else ("DP" if isinstance(current_dist, DataParallel) else "NONE")
        
        cache_key = f"torch_mesh_{device_mesh.shape}_{device_mesh.axis_names}_{dist_type}"
        cached = global_state.get_global_attribute(cache_key)
        if cached is not None:
            if get_device() == "meta" and cached.device_type != "meta":
                meta_cached = global_state.get_global_attribute(cache_key + "_meta", None)
                if meta_cached is not None:
                    return meta_cached
                meta_mesh = DeviceMesh("meta", cached.mesh, mesh_dim_names=cached.mesh_dim_names)
                global_state.set_global_attribute(cache_key + "_meta", meta_mesh)
                return meta_mesh
            return cached

        # CRITICAL FIX: If mesh is not yet created for this distribution, create it.
        # This ensures that convert_to_tensor() can always promote to DTensor
        # when a distribution is active.
        res_mesh = _to_backend_mesh(device_mesh)
        if get_device() == "meta" and res_mesh.device_type != "meta":
            meta_mesh = DeviceMesh("meta", res_mesh.mesh, mesh_dim_names=res_mesh.mesh_dim_names)
            return meta_mesh
        return res_mesh

    # Fallback to generic cached mesh if distributed is initialized
    if is_distributed:
        generic_cached = global_state.get_global_attribute("torch_device_mesh", None)
        if generic_cached is not None and hasattr(generic_cached, 'mesh'):
            if get_device() == "meta" and generic_cached.device_type != "meta":
                meta_cached = global_state.get_global_attribute("torch_device_mesh_meta", None)
                if meta_cached is not None:
                    return meta_cached
                meta_mesh = DeviceMesh("meta", generic_cached.mesh, mesh_dim_names=generic_cached.mesh_dim_names)
                global_state.set_global_attribute("torch_device_mesh_meta", meta_mesh)
                return meta_mesh
            return generic_cached
            
    return None


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
        
        res_mesh = cached
        if get_device() == "meta" and res_mesh.device_type != "meta":
            # Check for meta-cached version
            meta_cached = global_state.get_global_attribute(cache_key + "_meta", None)
            if meta_cached is not None:
                return meta_cached
            else:
                res_mesh = DeviceMesh("meta", res_mesh.mesh, mesh_dim_names=res_mesh.mesh_dim_names)
                global_state.set_global_attribute(cache_key + "_meta", res_mesh)
                return res_mesh
            
        global_state.set_global_attribute("torch_device_mesh", res_mesh)
        return res_mesh

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
                    
                    res_mesh = backend_mesh
                    if get_device() == "meta" and res_mesh.device_type != "meta":
                        res_mesh = DeviceMesh("meta", res_mesh.mesh, mesh_dim_names=res_mesh.mesh_dim_names)
                        global_state.set_global_attribute(cache_key + "_meta", res_mesh)
                    global_state.set_global_attribute("torch_device_mesh", res_mesh)
                    return res_mesh
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
                    
                    res_mesh = backend_mesh
                    if get_device() == "meta" and res_mesh.device_type != "meta":
                        res_mesh = DeviceMesh("meta", res_mesh.mesh, mesh_dim_names=res_mesh.mesh_dim_names)
                        global_state.set_global_attribute(cache_key + "_meta", res_mesh)
                    global_state.set_global_attribute("torch_device_mesh", res_mesh)
                    return res_mesh
    
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
    
    res_mesh = backend_mesh
    if get_device() == "meta" and res_mesh.device_type != "meta":
        res_mesh = DeviceMesh("meta", res_mesh.mesh, mesh_dim_names=res_mesh.mesh_dim_names)
        global_state.set_global_attribute(cache_key + "_meta", res_mesh)
    global_state.set_global_attribute("torch_device_mesh", res_mesh)
    return res_mesh


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

    # CRITICAL FIX: Ensure placements is always a list of Placement objects
    # Handle the case where placements might be a tuple (e.g., layout tuple passed directly)
    from keras.src.distribution.distribution_lib import TensorLayout
    if isinstance(placements, TensorLayout):
        placements = _layout_to_placements(placements.axes, tensor, device_mesh)
    elif placements is not None and not isinstance(placements, list):
        # Convert tuple or other iterables to list
        placements = list(placements)
    
    # Validate and fix placements - ensure each element is a Placement object
    if placements:
        from torch.distributed._tensor.placement_types import Placement
        safe_placements = []
        for p in placements:
            if isinstance(p, Placement):
                safe_placements.append(p)
            elif isinstance(p, Shard):
                safe_placements.append(p)
            elif isinstance(p, Replicate):
                safe_placements.append(p)
            elif isinstance(p, tuple):
                # CRITICAL FIX: If a tuple is passed as placement, convert it to Replicate
                # This can happen when layout tuples are mistakenly passed as placements
                safe_placements.append(Replicate())
            else:
                # Default to Replicate for unknown types
                safe_placements.append(Replicate())
        placements = safe_placements
    
    placements = [Replicate()] if placements is None else (placements if isinstance(placements, list) else [placements])
    # Validate placements against tensor ndim to avoid DTensor.from_local assertions
    safe_placements = []
    for p in placements:
        if isinstance(p, Shard):
            # Normalize negative dims
            dim = p.dim
            if dim < 0:
                dim = max(0, tensor.dim() + dim)
            # If shard dim is invalid for this tensor, fall back to Replicate
            if dim >= tensor.dim():
                safe_placements.append(Replicate())
            else:
                safe_placements.append(Shard(dim))
        else:
            safe_placements.append(p)

    return dtensor_from_local(tensor, device_mesh, safe_placements)


def dtensor_from_local(tensor, device_mesh, placements):
    """Safely create a DTensor from a local tensor, adjusting placements.

    This wrapper clamps or replaces invalid Shard dimensions with Replicate
    to avoid runtime assertions when parts of the code attempt to create
    a DTensor with a Shard on a non-existent tensor dimension.
    """
    if tensor is None:
        return tensor

    # CRITICAL FIX: Ensure placements is always a list of Placement objects
    # Handle the case where placements might be a tuple (e.g., layout tuple passed directly)
    from keras.src.distribution.distribution_lib import TensorLayout
    if isinstance(placements, TensorLayout):
        placements = _layout_to_placements(placements.axes, tensor, device_mesh)
    elif placements is not None and not isinstance(placements, list):
        # Convert tuple or other iterables to list
        placements = list(placements)
    
    # Validate and fix placements - ensure each element is a Placement object
    if placements:
        from torch.distributed._tensor.placement_types import Placement
        safe_placements = []
        for p in placements:
            if isinstance(p, Placement):
                safe_placements.append(p)
            elif isinstance(p, Shard):
                safe_placements.append(p)
            elif isinstance(p, Replicate):
                safe_placements.append(p)
            elif isinstance(p, tuple):
                # CRITICAL FIX: If a tuple is passed as placement, convert it to Replicate
                # This can happen when layout tuples are mistakenly passed as placements
                safe_placements.append(Replicate())
            else:
                # Default to Replicate for unknown types
                safe_placements.append(Replicate())
        placements = safe_placements
    
    placements = [Replicate()] if placements is None else (placements if isinstance(placements, list) else [placements])

    safe_placements = []
    for p in placements:
        if isinstance(p, Shard):
            dim = p.dim
            if dim < 0:
                dim = max(0, tensor.dim() + dim)
            if dim >= tensor.dim():
                safe_placements.append(Replicate())
            else:
                safe_placements.append(Shard(dim))
        else:
            safe_placements.append(p)

    try:
        # CRITICAL FIX: Ensure the local tensor is on the correct device for the mesh.
        # In multi-process mode, each rank sees its own GPU as cuda:0 (usually).
        # PyTorch DTensor.from_local requires the local tensor to be on the mesh's device.
        # EXCEPTION: "meta" tensors should stay on "meta" device during symbolic build.
        if tensor.device.type != "meta":
            if torch.cuda.is_available() and device_mesh.device_type == "cuda":
                # Map to the rank's specific assigned GPU (visible as 0 or mapped via current_device)
                local_device = torch.device(f"cuda:{torch.cuda.current_device()}")
                if tensor.device != local_device:
                    tensor = tensor.to(local_device)
            elif device_mesh.device_type == "cpu" and tensor.device.type != "cpu":
                tensor = tensor.to("cpu")

        return DTensor.from_local(tensor, device_mesh, safe_placements)
    except AssertionError as e:
        # As a last-resort fallback, replace any remaining Shard with Replicate
        # and retry. This should avoid the "Sharding dim > tensor.ndim" assertion.
        repl = [Replicate() for _ in safe_placements]
        return DTensor.from_local(tensor, device_mesh, repl)


def is_dtensor(tensor):
    """Check if a tensor is a DTensor.
    
    This function uses both isinstance check and duck-typing for reliable detection.
    Duck-typing is used as a fallback because DTensor might be imported from different
    modules in some cases.
    """
    if tensor is None:
        return False
        
    # Handle torch.nn.Parameter by checking its underlying data
    if isinstance(tensor, torch.nn.Parameter):
        return is_dtensor(tensor.data)
        
    # First try isinstance check (most reliable if DTensor is imported correctly)
    try:
        if isinstance(tensor, DTensor):
            return True
    except TypeError:
        # DTensor might not be importable in some contexts
        pass
    
    # Duck-typing fallback: DTensor has these distinctive methods/attributes
    # that regular tensors don't have
    if hasattr(tensor, 'to_local') and hasattr(tensor, 'placements') and hasattr(tensor, 'device_mesh'):
        # Additional check: to_local and redistribute should be present
        if callable(getattr(tensor, 'to_local', None)) and callable(getattr(tensor, 'redistribute', None)):
            return True
    
    return False


def get_dtensor_mesh(tensor):
    """Get the device mesh from a DTensor or Parameter wrapping a DTensor."""
    if tensor is None:
        return None
    if isinstance(tensor, torch.nn.Parameter):
        return get_dtensor_mesh(tensor.data)
    return getattr(tensor, 'device_mesh', None)


def get_dtensor_placements(tensor):
    """Get the placements from a DTensor or Parameter wrapping a DTensor."""
    if tensor is None:
        return None
    if isinstance(tensor, torch.nn.Parameter):
        return get_dtensor_placements(tensor.data)
    return getattr(tensor, 'placements', None)


def dtensor_to_local(tensor):
    """Convert DTensor to local tensor format."""
    if tensor is None:
        return tensor

    # Use the improved is_dtensor function for reliable detection
    if is_dtensor(tensor):
        if isinstance(tensor, torch.nn.Parameter):
            return tensor.data.to_local()
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


class _AllReduceWithGradient(torch.autograd.Function):
    """Custom autograd function for all-reduce with proper gradient flow.
    
    This is needed for ModelParallel training where the output has Partial placement
    and needs to be all-reduced across ranks. Using plain torch.distributed.all_reduce
    breaks the autograd graph and causes gradient shape mismatches.
    """

    @staticmethod
    def forward(ctx, tensor):
        output = tensor.clone()
        torch.distributed.all_reduce(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # For all-reduce, the gradient should be the same shape as output
        # and should be all-reduced again (or just kept as-is since each rank
        # gets the full gradient)
        # Actually, for proper gradient flow in all-reduce, we just return
        # the gradient as-is since each rank computes gradients locally
        return grad_output


def _all_reduce_with_grad(tensor):
    """Perform all-reduce with proper gradient flow for ModelParallel."""
    return _AllReduceWithGradient.apply(tensor)


def _convert_structure(x, device_mesh=None, to_dtensor=True, gather_sharded=True):
    """Unified recursive structure converter for DTensor operations."""
    if x is None:
        return x

    # Debug info
    debug_mode = os.environ.get("KERAS_DISTRIBUTION_DEBUG", "0") == "1"
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    if debug_mode:
        print(f"DEBUG | [Rank {rank}] _convert_structure called: type(x)={type(x).__name__}, to_dtensor={to_dtensor}, x_is_dtensor={is_dtensor(x)}")

    if is_dtensor(x):
        if not to_dtensor:
            # Unwrap Parameter if needed
            x_data = x.data if isinstance(x, torch.nn.Parameter) else x
            
            if gather_sharded and not all(isinstance(p, Replicate) for p in x_data.placements):
                if torch.distributed.is_initialized():
                    shard_dim = next((i for i, p in enumerate(x_data.placements) if isinstance(p, Shard)), None)
                    if shard_dim is not None:
                        local_tensor = x_data.to_local()
                        if local_tensor.requires_grad:
                            return _all_gather_with_grad(local_tensor, shard_dim)
                        else:
                            output = [torch.empty_like(local_tensor) for _ in range(torch.distributed.get_world_size())]
                            torch.distributed.all_gather(output, local_tensor.contiguous())
                            return torch.cat(output, dim=shard_dim)
            return x_data.to_local()
        return x

    if isinstance(x, (torch.Tensor, np.ndarray)):
        # CRITICAL FIX: Automatically detect if we should convert to DTensor
        # even when to_dtensor is not explicitly set to True.
        # This handles internal tensors (like causal masks) that are created
        # inside layer operations and need to be DTensors when a DeviceMesh is active.
        
        # First, determine if we should convert to DTensor
        should_convert = to_dtensor
        
        # If to_dtensor is False, check if we have an active DeviceMesh
        # that would require DTensor conversion for proper operation
        if not should_convert:
            # Check if there's an active DeviceMesh via multiple methods
            from keras.src.distribution.distribution_lib import distribution, ModelParallel
            
            current_dist = distribution()
            is_mp = isinstance(current_dist, ModelParallel) if current_dist else False
            
            # Check cached MP multi-process state - this is the key fix!
            # Even if distribution() returns None (scope exited), we should still
            # check the cached state from TorchTrainer._cache_mp_multi_process_state()
            global _MP_MULTI_PROCESS_STATE
            cached_mp_state = _MP_MULTI_PROCESS_STATE
            
            # Also check if torch distributed is initialized with a 1D mesh
            # (which indicates ModelParallel in multi-process mode)
            is_distributed = torch.distributed.is_initialized()
            
            # Get device mesh from various sources
            torch_device_mesh = None
            
            # Try current distribution first
            if current_dist is not None and hasattr(current_dist, 'device_mesh'):
                torch_device_mesh = _to_backend_mesh(current_dist.device_mesh)
            
            # Fallback to cached mesh
            if torch_device_mesh is None:
                torch_device_mesh = _get_default_device_mesh()
            
            # If we have a mesh and distributed is active, we need to convert
            # OR if the cached MP multi-process state is True, we must convert
            if torch_device_mesh is not None and (is_distributed or cached_mp_state):
                should_convert = True
            
            if debug_mode and should_convert:
                print(f"DEBUG | [Rank {rank}] _convert_structure: Auto-detected need to convert to DTensor (mesh={torch_device_mesh is not None}, dist={is_distributed}, mp={is_mp}, cached_mp={cached_mp_state})")
        
        if should_convert:
            # Identify the backend mesh
            from keras.src.distribution.distribution_lib import distribution
            current_dist = distribution()
            torch_device_mesh = None
            
            if current_dist is not None and hasattr(current_dist, 'device_mesh'):
                torch_device_mesh = _to_backend_mesh(current_dist.device_mesh)
            elif device_mesh is not None:
                torch_device_mesh = device_mesh if hasattr(device_mesh, 'mesh') else _to_backend_mesh(device_mesh)
            
            # Fallback to get default mesh if still None
            if torch_device_mesh is None:
                torch_device_mesh = _get_default_device_mesh()

            # CRITICAL FIX: If a mesh exists and we are distributed, we MUST return a DTensor.
            # Mixed tensors cause crashes in PyTorch distributed operators.
            if torch_device_mesh is not None and torch.distributed.is_initialized():
                if debug_mode:
                    print(f"DEBUG | [Rank {rank}] _convert_structure: Promoting {type(x)} to DTensor")
                
                # Convert numpy arrays or other types to torch tensor first
                if isinstance(x, np.ndarray):
                    if x.dtype == np.uint32: x = x.astype(np.int64)
                    dtype = standardize_dtype(x.dtype)
                    if dtype == "bfloat16": 
                        x = x.astype(np.float32)
                        dtype = "bfloat16"
                    local_tensor = torch.as_tensor(x, dtype=to_torch_dtype(dtype), device=get_device())
                else:
                    local_tensor = x
                
                # Inputs for ModelParallel must be Replicated across the mesh
                placements = [Replicate()] * torch_device_mesh.mesh.ndim
                return dtensor_from_local(local_tensor, torch_device_mesh, placements)

        # Fallback for non-distributed (CPU / Single-process)
        if isinstance(x, np.ndarray):
            return convert_to_tensor(x)
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
    
    # CRITICAL FIX: For ModelParallel in multi-process mode, convert inputs to DTensors
    # with REPLICATE placement. Each rank needs the full input because model weights
    # are sharded across the "model" axis, and each rank computes with its portion
    # of the sharded weights.
    #
    # The key difference from single-process:
    # - In single-process MP: inputs can be on different devices, need proper device placement
    # - In multi-process MP: each process has its own GPU, inputs should be Replicated DTensors
    #
    # We must convert inputs to DTensors with Replicate() placement, NOT skip conversion.
    if (is_mp or cached_mp_multi_process) and is_distributed:
        if debug_mode:
            rank = 0
            try:
                import torch.distributed as dist
                if dist.is_available() and dist.is_initialized():
                    rank = dist.get_rank()
            except:
                pass
            print(f"DEBUG | [Rank {rank}] prepare_input_for_distribution: Converting to DTensor with Replicate placement for ModelParallel in multi-process mode (is_mp={is_mp}, cached_mp_multi_process={cached_mp_multi_process})")
        
        # Get the device mesh for DTensor conversion
        torch_device_mesh = None
        
        # Try to get the mesh from current distribution first
        if current_dist is not None and hasattr(current_dist, 'device_mesh'):
            torch_device_mesh = _to_backend_mesh(current_dist.device_mesh)
        
        # Fallback to cached mesh if needed
        if torch_device_mesh is None:
            torch_device_mesh = _get_default_device_mesh()
        
        if torch_device_mesh is not None:
            # CRITICAL FIX: Convert input to DTensor with REPLICATE placement
            # This ensures the input is a DTensor so operations like token_emb + position_emb work
            # The input is replicated (each rank has full input), not sharded
            return _convert_structure(x, torch_device_mesh, to_dtensor=True, gather_sharded=False)
        
        # If no mesh available, just ensure tensor is on correct device and return
        if isinstance(x, torch.Tensor):
            local_rank = 0
            try:
                import torch.distributed as dist
                if dist.is_available() and dist.is_initialized():
                    local_rank = dist.get_rank()
            except:
                pass
            
            if torch.cuda.is_available():
                current_device = torch.cuda.current_device()
                if x.is_cuda:
                    if x.device.index != current_device:
                        x = x.to(f"cuda:{current_device}")
                else:
                    x = x.to(device=f"cuda:{local_rank}")
        elif isinstance(x, np.ndarray):
            if torch.cuda.is_available():
                local_rank = 0
                try:
                    import torch.distributed as dist
                    if dist.is_available() and dist.is_initialized():
                        local_rank = dist.get_rank()
                except:
                    pass
                
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
    
    IMPORTANT: In ModelParallel multi-process mode:
    - y_pred (model outputs) are either:
      - DTensors with sharded placements (needs all-gather)
      - Local tensors with sharded shape from DTensor matmul fallback
    - y (labels) are local tensors with FULL shape (should NOT be all-gathered)
    
    This function all-gathers ONLY when the tensor is a DTensor with sharded placements.
    Plain local tensors (like labels) are returned as-is to preserve correct shape.
    
    CRITICAL FIX: In ModelParallel multi-process mode, skip this function entirely
    to avoid issues with DTensor handling. The trainer should not call this function
    in MP multi-process mode.
    
    CRITICAL FIX 2: This function should ONLY process y_pred (model outputs),
    NOT y (labels). Labels are local tensors and should never be processed.
    If x is NOT a DTensor, return it as-is immediately.
    """
    from keras.src.distribution.distribution_lib import distribution, ModelParallel
    
    # CRITICAL FIX: If x is not a DTensor, return it as-is immediately
    # This handles the case where y (labels) are passed to this function
    # Labels are always local tensors and should never be converted
    # Use is_dtensor for reliable detection
    if not is_dtensor(x):
        return x
    
    # Check if we have an active ModelParallel distribution
    current_dist = distribution()
    is_mp = isinstance(current_dist, ModelParallel)
    
    # Also check the cached MP multi-process state.
    global _MP_MULTI_PROCESS_STATE
    cached_mp_state = _MP_MULTI_PROCESS_STATE
    
    # Debug: print the global state
    debug_mode = os.environ.get("KERAS_DISTRIBUTION_DEBUG", "0") == "1"
    if debug_mode:
        rank = 0
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
        except:
            pass
        print(f"DEBUG | [Rank {rank}] prepare_output_for_loss START: cached_mp_state={cached_mp_state}, x_type={type(x).__name__}, x_shape={getattr(x, 'shape', 'N/A')}")
    
    # CRITICAL FIX: In MP multi-process mode, we need to convert DTensor to local tensor
    # Returning DTensor as-is causes "aten.sub.Tensor: got mixed torch.Tensor and DTensor" 
    # errors during loss computation because PyTorch can't mix DTensors with regular tensors.
    # NOTE: previously we returned early here for cached MP multi-process state
    # which converted DTensor outputs to local shards. That caused label/output
    # shape mismatches when computing the loss (labels are full local tensors).
    # Instead, fall through to the standard DTensor handling below which will
    # perform an all-gather or all-reduce when needed. We keep the debug
    # message for visibility but DO NOT return early.
    if cached_mp_state and torch.distributed.is_initialized() and debug_mode:
        print(f"DEBUG | [Rank {rank}] prepare_output_for_loss: MP multi-process mode active (cached), proceeding to DTensor handling")
    
    # Even if the scope has exited, check if we have a cached ModelParallel mesh
    # or if we cached the MP multi-process state
    if not is_mp and torch.distributed.is_initialized():
        if cached_mp_state:
            is_mp = True
        else:
            # Fallback: check if there's a cached MP mesh
            cached_mesh = global_state.get_global_attribute("torch_device_mesh", None)
            if cached_mesh is not None and hasattr(cached_mesh, 'mesh'):
                if cached_mesh.mesh.ndim == 1:
                    is_mp = True
    
    if debug_mode:
        rank = 0
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
        except:
            pass
        print(f"DEBUG | [Rank {rank}] prepare_output_for_loss AFTER CHECK: is_mp={is_mp}, cached_mp_state={cached_mp_state}, dist_init={torch.distributed.is_initialized()}")
    
    # If not MP mode, just convert DTensor to local tensor if needed
    if not is_mp:
        if is_dtensor(x):
            if isinstance(x, torch.nn.Parameter):
                return x.data.to_local()
            return x.to_local()
        return x
    
    # For ModelParallel: 
    # - y_pred: DTensor with sharded placements -> all-gather
    # - y: local tensor with full shape -> return as-is (DO NOT all-gather!)
    
    # CRITICAL FIX: Check for DTensor using duck typing in case of module mismatch
    # Sometimes DTensor can be imported from different modules
    # Use different variable name to avoid shadowing the module-level is_dtensor function
    x_is_dtensor = is_dtensor(x)
    
    if debug_mode:
        rank = 0
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
        except:
            pass
        print(f"DEBUG | [Rank {rank}] prepare_output_for_loss: is_dtensor={x_is_dtensor}, x_type={type(x).__name__}")
    
    # Check if x is a DTensor - this is the model output that needs all-gather
    if x_is_dtensor:
        # Unwrap data if it's a Parameter
        x_data = x.data if isinstance(x, torch.nn.Parameter) else x
        
        # Debug: print the placements and local shape
        if debug_mode:
            rank = 0
            try:
                import torch.distributed as dist
                if dist.is_available() and dist.is_initialized():
                    rank = dist.get_rank()
            except:
                pass
            local_shape = x_data.to_local().shape
            print(f"DEBUG | [Rank {rank}] prepare_output_for_loss: DTensor placements = {x_data.placements}, local_shape = {local_shape}, global_shape = {x_data.shape}")
        
        # Get local tensor shape to determine if sharding is applied
        local_tensor = x_data.to_local()
        local_shape = local_tensor.shape
        
        # Check if this is a sharded DTensor by comparing local vs global shape
        # OR if it has Partial(sum) placement which requires all-reduce
        global_shape = x_data.shape
        
        # Check for Partial placement - this indicates the output needs all-reduce
        # (e.g., from ColwiseParallel/RowwiseParallel in tensor parallelism)
        # Use the public API from torch.distributed._tensor.placement_types
        from torch.distributed._tensor.placement_types import Partial
        has_partial = any(isinstance(p, Partial) for p in x_data.placements)
        
        is_sharded = (len(local_shape) > 0 and len(global_shape) > 0 and 
                      local_shape[-1] != global_shape[-1])
        
        # Need all-gather/reduce if:
        # 1. Tensor is sharded (local shape != global shape on some dimension)
        # 2. OR tensor has Partial placement (needs all-reduce for proper output)
        needs_all_gather = is_sharded or has_partial
        
        if debug_mode:
            print(f"DEBUG | [Rank {rank}] prepare_output_for_loss: is_sharded = {is_sharded}, has_partial = {has_partial}, needs_all_gather = {needs_all_gather} (local={local_shape}, global={global_shape})")
        
        if needs_all_gather:
            # Check if it's Partial placement - need to all-reduce instead of all-gather
            if has_partial:
                # For Partial placement, we need to all-reduce to get the correct output
                # This handles the case where ColwiseParallel/RowwiseParallel produces
                # Partial(sum) output that needs to be summed across all ranks
                if debug_mode:
                    print(f"DEBUG | [Rank {rank}] prepare_output_for_loss: all-reducing DTensor with Partial placement")
                
                # CRITICAL FIX: Use custom autograd function to preserve gradients
                # Using plain torch.distributed.all_reduce breaks the autograd graph
                # and causes gradient shape mismatches in optimizer
                return _all_reduce_with_grad(local_tensor)
            
            # For Shard placement - all-gather the tensor
            try:
                # Determine shard dimension from placements or infer from shape difference
                shard_dim = None
                for i, p in enumerate(x_data.placements):
                    if isinstance(p, Shard):
                        shard_dim = p.dim
                        break
                
                if shard_dim is None:
                    # Infer from shape difference
                    for i in range(len(local_shape)):
                        if i < len(global_shape) and local_shape[i] != global_shape[i]:
                            shard_dim = i
                            break
                
                if shard_dim is None:
                    shard_dim = -1  # Default to last dimension
                
                if debug_mode:
                    print(f"DEBUG | [Rank {rank}] prepare_output_for_loss: all-gathering local_tensor shape {local_shape} on dim {shard_dim}")
                
                world_size = torch.distributed.get_world_size()
                # All-gather with proper gradient flow
                if local_tensor.requires_grad:
                    return _all_gather_with_grad(local_tensor, shard_dim)
                else:
                    output = [torch.empty_like(local_tensor) for _ in range(world_size)]
                    torch.distributed.all_gather(output, local_tensor.contiguous())
                    return torch.cat(output, dim=shard_dim)
            except Exception as e:
                if debug_mode:
                    print(f"DEBUG | [Rank {rank}] prepare_output_for_loss: all-gather failed: {e}")
                return local_tensor
        else:
            # DTensor with same local and global shape (replicated) - just get local tensor
            return local_tensor
    
    
    
    # Not a DTensor - this is likely labels (y) which are full local tensors
    # DO NOT all-gather! Labels are full local data, not sharded outputs.
    return x
