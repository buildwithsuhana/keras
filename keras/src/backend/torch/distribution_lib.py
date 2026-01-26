import torch
import torch.distributed as dist
import numpy as np
import re
from typing import Optional, Tuple, Union, Any

# Try to import DTensor (optional - not required for native sharding)
try:
    from torch.distributed._tensor import DTensor, Replicate, Shard
    from torch.distributed.device_mesh import DeviceMesh as TorchDeviceMesh
    from torch.distributed._tensor import distribute_tensor as torch_distribute_tensor
    DTENSOR_AVAILABLE = True
except ImportError:
    DTENSOR_AVAILABLE = False
    DTensor = None
    Replicate = None
    Shard = None
    TorchDeviceMesh = None
    torch_distribute_tensor = None

from keras.src.backend.torch.core import convert_to_tensor


def is_dtensor_available():
    """Check if PyTorch DTensor is available."""
    return DTENSOR_AVAILABLE


def _is_dtensor(tensor):
    """Check if a tensor is a PyTorch DTensor."""
    if not DTENSOR_AVAILABLE:
        return False
    return isinstance(tensor, DTensor)


def _get_current_rank() -> int:
    """Get the current process rank."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def _get_world_size() -> int:
    """Get the total number of processes."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def _get_local_rank() -> int:
    """Get the local process rank."""
    if dist.is_initialized():
        return dist.get_rank() % torch.cuda.device_count() if torch.cuda.is_available() else 0
    return 0


def _get_shard_info(layout) -> Tuple[Optional[str], Optional[int]]:
    """Extract sharding axis name and index from a Keras TensorLayout.
    
    Args:
        layout: A Keras TensorLayout or None
        
    Returns:
        Tuple of (axis_name, axis_index) or (None, None) if no sharding
    """
    if layout is None:
        return None, None
    
    from keras.src.distribution.distribution_lib import TensorLayout
    
    if not isinstance(layout, TensorLayout):
        return None, None
    
    # Find the first non-None axis (sharding dimension)
    axes = layout.axes
    device_mesh = layout.device_mesh
    
    if device_mesh is None:
        return None, None
    
    axis_names = device_mesh.axis_names
    
    for i, axis in enumerate(axes):
        if axis is not None:
            return axis, i
    
    return None, None


def _shard_tensor_native(tensor, mesh_axis: int, num_shards: int, rank: int) -> torch.Tensor:
    """Shard a tensor along a dimension using native PyTorch operations.
    
    This is the DTensor-free alternative for tensor sharding.
    
    Args:
        tensor: The tensor to shard
        mesh_axis: The axis in the device mesh to shard along
        num_shards: Number of shards to create
        rank: Current process rank
        
    Returns:
        The shard of the tensor for this rank
    """
    if tensor.dim() == 0:
        # Scalar tensor - replicate across all ranks
        return tensor
    
    # Calculate chunk size
    dim_size = tensor.size(mesh_axis)
    # Ensure chunk_size is a native Python int (not numpy.int64 or torch scalar)
    # torch.split() requires Python ints for split_size_or_sections
    base_chunk_size = int(dim_size // num_shards)
    base_chunk_size = max(1, base_chunk_size)
    
    # Handle partial chunks
    if rank == num_shards - 1:
        # Last rank gets the remainder
        chunk_size = int(dim_size) - base_chunk_size * (num_shards - 1)
    else:
        chunk_size = base_chunk_size
    
    # Split the tensor
    shards = torch.split(tensor, chunk_size, dim=mesh_axis)
    return shards[rank]


def _gather_tensors_native(tensor, mesh_axis: int, num_shards: int) -> torch.Tensor:
    """Gather sharded tensors back into a full tensor.
    
    Args:
        tensor: The local shard
        mesh_axis: The axis along which tensors were sharded
        num_shards: Number of shards
        
    Returns:
        The gathered full tensor
    """
    if num_shards == 1:
        return tensor
    
    if not dist.is_initialized():
        return tensor
    
    # Use all_gather_object for simple case, or all_gather for tensors
    world_size = dist.get_world_size()
    
    # Get tensor shape
    local_shape = list(tensor.shape)
    local_shape[mesh_axis] = local_shape[mesh_axis] * world_size
    
    # Create output tensor on the current device
    output = torch.empty(local_shape, dtype=tensor.dtype, device=tensor.device)
    
    # Use all_gather
    tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor)
    
    # Concatenate along the shard dimension
    return torch.cat(tensor_list, dim=mesh_axis)


def _get_dtensor_mesh_and_placements(layout):
    """Extract torch mesh and placements from a Keras TensorLayout.
    
    Args:
        layout: A Keras TensorLayout or None
        
    Returns:
        Tuple of (torch_mesh, placements) or (None, None) if layout is None
    """
    if layout is None:
        return None, None
    
    from keras.src.distribution.distribution_lib import TensorLayout
    
    if not isinstance(layout, TensorLayout):
        return None, None
    
    try:
        torch_mesh = layout.device_mesh.backend_mesh
        placements = layout.backend_layout
        return torch_mesh, placements
    except AttributeError:
        return None, None


def _convert_to_dtensor(tensor, mesh, placements):
    """Convert a regular torch.Tensor to DTensor.
    
    This function ensures that when operating with DTensors, all operands
    are DTensors. If the input is already a DTensor, it is returned as-is.
    If it's a regular tensor, it is converted to a DTensor with the given
    mesh and placements.
    
    Args:
        tensor: A torch.Tensor or DTensor
        mesh: The DeviceMesh to use for distribution
        placements: The placements for the DTensor
        
    Returns:
        A DTensor representation of the input
    """
    if not DTENSOR_AVAILABLE:
        return tensor
    
    if _is_dtensor(tensor):
        return tensor
    
    if mesh is None or placements is None:
        return tensor
    
    return torch_distribute_tensor(tensor, mesh, placements)


def _convert_batch_to_dtensor(batch, mesh, placements):
    """Convert all tensors in a batch to DTensors.

    This function handles nested structures (tuples, lists, dicts) and
    converts all torch.Tensor elements to DTensors.

    IMPORTANT: This function uses recursion instead of tree.map_structure
    because tree.map_structure internally uses PyTorch's tree_map which
    fails when processing DTensors.
    """
    if not DTENSOR_AVAILABLE:
        return batch
    
    if mesh is None or placements is None:
        return batch

    def convert_single(x):
        if _is_dtensor(x):
            return x
        if isinstance(x, torch.Tensor):
            return torch_distribute_tensor(x, mesh, placements)
        return x

    def _convert_recursive(item):
        """Recursively convert tensors without using tree_map."""
        if _is_dtensor(item):
            return item
        if isinstance(item, torch.Tensor):
            return torch_distribute_tensor(item, mesh, placements)
        elif isinstance(item, tuple):
            return tuple(_convert_recursive(x) for x in item)
        elif isinstance(item, list):
            return [_convert_recursive(x) for x in item]
        elif isinstance(item, dict):
            return {k: _convert_recursive(v) for k, v in item.items()}
        else:
            return item

    return _convert_recursive(batch)


def _ensure_dtensor(tensor, mesh, placements=None):
    """Convert a regular torch.Tensor to DTensor if needed.

    This function ensures that when operating with DTensors, all operands
    are DTensors. If the input is already a DTensor, it is returned as-is.
    If it's a regular tensor, it is converted to a DTensor with the given
    mesh and placements.

    Args:
        tensor: A torch.Tensor or DTensor
        mesh: The DeviceMesh to use for distribution
        placements: The placements for the DTensor. If None, uses Replicate.

    Returns:
        A DTensor representation of the input
    """
    if not DTENSOR_AVAILABLE:
        return tensor
    
    if _is_dtensor(tensor):
        return tensor

    # Regular tensor that needs to be converted to DTensor
    if placements is None:
        placements = (Replicate(),)

    return torch_distribute_tensor(tensor, mesh, placements)


def _convert_to_matching_dtensor(tensor, dtensor):
    """Convert a regular tensor to a DTensor matching the reference DTensor.

    This function is used to ensure that when performing operations between
    a DTensor and a regular tensor, the regular tensor is converted to a
    DTensor with the same mesh and placements as the reference DTensor.

    Args:
        tensor: A torch.Tensor or DTensor
        dtensor: A DTensor to get mesh and placements from

    Returns:
        A DTensor with the same mesh and placements as dtensor
    """
    if not DTENSOR_AVAILABLE:
        return tensor
    
    if _is_dtensor(tensor):
        return tensor

    return torch_distribute_tensor(tensor, dtensor.device_mesh, dtensor.placements)


def list_devices(device_type=None):
    """List available local devices."""
    device_type = device_type or "cuda"
    if device_type == "cuda":
        return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    return ["cpu"]


def get_device_count(device_type=None):
    """Return the total number of devices in the cluster."""
    if dist.is_initialized():
        return dist.get_world_size()
    return torch.cuda.device_count()


def initialize(job_addresses=None, num_processes=None, process_id=None):
    """Initialize the distributed process group for Torch SPMD."""
    if not dist.is_initialized():
        # SPMD usually utilizes the NCCL backend for GPU coordination
        dist.init_process_group(backend="nccl")

    if torch.cuda.is_available():
        # Map the process to a specific local physical GPU based on rank
        local_rank = _get_local_rank()
        torch.cuda.set_device(local_rank)


def _to_backend_mesh(device_mesh):
    """Bridge for DeviceMesh.backend_mesh"""
    if not DTENSOR_AVAILABLE:
        return None
    
    # Your framework signature: DeviceMesh(device_type, mesh, mesh_dim_names=...)
    # mesh can be a tuple (shape) or a list of ranks.

    if device_mesh.devices is not None:
        # Convert device names to integer device indices
        # device_mesh.devices can be np.array(['cuda:0', 'cuda:1']) or similar
        device_ids = []
        for d in device_mesh.devices:
            if isinstance(d, str):
                # Extract device index from string like 'cuda:0' -> 0
                res = re.search(r'\d+', d)
                device_ids.append(int(res.group()) if res else 0)
            else:
                device_ids.append(int(d))
    else:
        # Fallback to shape if devices is None
        device_ids = device_mesh.shape

    # Use positional arguments as required by PyTorch's DeviceMesh
    return TorchDeviceMesh(
        "cuda" if torch.cuda.is_available() else "cpu",
        device_ids,
        mesh_dim_names=device_mesh.axis_names
    )


def _to_backend_layout(layout):
    """Convert Keras Layout to Torch placements."""
    if not DTENSOR_AVAILABLE:
        return None
    
    torch_mesh = layout.device_mesh.backend_mesh
    placements = []
    for axis in layout.axes:
        if axis is None:
            placements.append(Replicate())
        else:
            # Find the index of the axis name in the mesh dimension names
            dim_index = list(torch_mesh.mesh_dim_names).index(axis)
            placements.append(Shard(dim_index))
    return placements


def distribute_variable(variable, layout, use_dtensor=None):
    """Distribute/shard a variable across devices.
    
    This function supports both DTensor-based distribution and native
    PyTorch sharding. By default, it uses native PyTorch sharding for
    better compatibility with torch.compile.
    
    Args:
        variable: Either a Keras Variable object or a torch.Tensor
        layout: The TensorLayout specifying how to shard the variable
        use_dtensor: If True, force DTensor usage. If False, use native sharding.
                    If None (default), use native sharding unless DTensor is
                    explicitly required.
        
    Returns:
        A torch.nn.Parameter containing the sharded tensor
    """
    rank = _get_current_rank()
    
    # Check if sharding is needed
    if layout is None:
        # No layout specified - just return as regular tensor
        if hasattr(variable, 'value'):
            tensor_value = variable.value
        else:
            tensor_value = variable
        
        if not isinstance(tensor_value, torch.Tensor):
            tensor_value = convert_to_tensor(tensor_value)
        
        return torch.nn.Parameter(tensor_value)
    
    # Determine sharding axis and device mesh info
    axis_name, axis_idx = _get_shard_info(layout)
    
    if axis_name is None:
        # No sharding specified - replicate
        if hasattr(variable, 'value'):
            tensor_value = variable.value
        else:
            tensor_value = variable
        
        if not isinstance(tensor_value, torch.Tensor):
            tensor_value = convert_to_tensor(tensor_value)
        
        return torch.nn.Parameter(tensor_value)
    
    # Get device mesh info
    device_mesh = layout.device_mesh
    num_devices = np.prod(device_mesh.shape)
    world_size = _get_world_size()
    
    # Determine number of shards
    num_shards = min(num_devices, world_size)
    
    # Decide on distribution method
    force_dtensor = use_dtensor is True
    
    if force_dtensor and DTENSOR_AVAILABLE:
        # Use DTensor-based distribution
        if hasattr(variable, 'value'):
            tensor_value = variable.value
        else:
            tensor_value = variable
        
        # Distribute using DTensor
        torch_mesh = layout.device_mesh.backend_mesh
        placements = layout.backend_layout
        sharded_tensor = torch_distribute_tensor(tensor_value, torch_mesh, placements)
        
        if rank == 0:
            print(f"[BACKEND] Sharding variable with DTensor. Axis: {axis_name}, Placements: {placements}")
        
        return torch.nn.Parameter(sharded_tensor)
    
    # Use native PyTorch sharding (default)
    if hasattr(variable, 'value'):
        tensor_value = variable.value
    else:
        tensor_value = variable
    
    if not isinstance(tensor_value, torch.Tensor):
        tensor_value = convert_to_tensor(tensor_value)
    
    # Get this rank's shard
    local_rank = rank % num_shards
    sharded_tensor = _shard_tensor_native(tensor_value, axis_idx, num_shards, local_rank)
    
    if rank == 0:
        print(f"[BACKEND] Sharding variable natively. Axis: {axis_name} (dim={axis_idx}), "
              f"Num shards: {num_shards}, Local rank: {local_rank}")
    
    return torch.nn.Parameter(sharded_tensor)


def distribute_tensor(value, layout, use_dtensor=None):
    """The core engine for Model Parallelism.
    
    This function supports both DTensor-based distribution and native
    PyTorch sharding. By default, it uses native PyTorch sharding for
    better compatibility with torch.compile.
    
    Args:
        value: The tensor to distribute
        layout: The TensorLayout specifying how to distribute
        use_dtensor: If True, force DTensor usage. If False, use native sharding.
                    If None (default), use native sharding.
        
    Returns:
        The distributed tensor
    """
    from keras.src.distribution.distribution_lib import TensorLayout

    if not isinstance(layout, TensorLayout):
        return value
    
    # Determine sharding axis
    axis_name, axis_idx = _get_shard_info(layout)
    
    if axis_name is None:
        # No sharding needed
        if not isinstance(value, torch.Tensor):
            value = convert_to_tensor(value)
        return value
    
    # Get device mesh info
    device_mesh = layout.device_mesh
    num_devices = np.prod(device_mesh.shape)
    world_size = _get_world_size()
    
    # Determine number of shards
    num_shards = min(num_devices, world_size)
    
    # Decide on distribution method
    force_dtensor = use_dtensor is True
    
    if force_dtensor and DTENSOR_AVAILABLE:
        # Use DTensor-based distribution
        if not isinstance(value, torch.Tensor):
            value = convert_to_tensor(value)
        
        torch_mesh = layout.device_mesh.backend_mesh
        placements = layout.backend_layout
        return torch_distribute_tensor(value, torch_mesh, placements)
    
    # Use native PyTorch sharding (default)
    if not isinstance(value, torch.Tensor):
        value = convert_to_tensor(value)
    
    rank = _get_current_rank()
    local_rank = rank % num_shards
    return _shard_tensor_native(value, axis_idx, num_shards, local_rank)


def distribute_data_input(data, layout, batch_dim_name=None, use_dtensor=None):
    """Distribute input data batches across devices.
    
    This function supports both DTensor-based distribution and native
    PyTorch sharding. By default, it uses native PyTorch sharding.
    
    Args:
        data: Input data tensor (torch.Tensor or numpy array)
        layout: The TensorLayout specifying how to distribute the data
        batch_dim_name: Optional name of the batch dimension for data parallelism
        use_dtensor: If True, force DTensor usage. If False, use native sharding.
                    If None (default), use native sharding.
        
    Returns:
        The distributed input data
    """
    from keras.src.distribution.distribution_lib import TensorLayout
    from keras.src.backend.torch.core import convert_to_tensor

    rank = _get_current_rank()
    
    # Convert to torch tensor first if needed
    if not isinstance(data, torch.Tensor):
        data = convert_to_tensor(data)
    
    # Handle DTensor input - return as-is
    if _is_dtensor(data):
        return data
    
    # If no layout provided, return data as-is (no distribution)
    if layout is None:
        return data
    
    # If layout is provided, distribute the tensor
    if isinstance(layout, TensorLayout):
        # Determine sharding axis
        axis_name, axis_idx = _get_shard_info(layout)
        
        # Get device mesh info
        device_mesh = layout.device_mesh
        num_devices = np.prod(device_mesh.shape)
        world_size = _get_world_size()
        
        # Determine number of shards
        num_shards = min(num_devices, world_size)
        
        # Decide on distribution method
        force_dtensor = use_dtensor is True
        
        if force_dtensor and DTENSOR_AVAILABLE:
            # Use DTensor-based distribution
            torch_mesh = layout.device_mesh.backend_mesh
            placements = layout.backend_layout
            
            # Check if there's a batch dimension to distribute along
            if batch_dim_name is not None and batch_dim_name in layout.device_mesh.axis_names:
                # Find the batch dimension index
                batch_dim_idx = list(layout.device_mesh.axis_names).index(batch_dim_name)
                # Update placements to shard along batch dimension
                placements = list(placements)
                placements[batch_dim_idx] = Shard(batch_dim_idx)
                placements = tuple(placements)

            if hasattr(data, "shape"):
                print(f"[Rank {rank}] Distributing input batch with DTensor. Local shape: {data.shape}, "
                      f"placements: {placements}")

            return torch_distribute_tensor(data, torch_mesh, placements)
        
        # Use native PyTorch sharding (default)
        # Check if batch dimension sharding is requested
        if batch_dim_name is not None and batch_dim_name in device_mesh.axis_names:
            axis_idx = list(device_mesh.axis_names).index(batch_dim_name)
        
        if hasattr(data, "shape"):
            print(f"[Rank {rank}] Distributing input batch natively. Local shape: {data.shape}, "
                  f"shard dim: {axis_idx}, num_shards: {num_shards}")
        
        # Shard the data
        local_rank = rank % num_shards
        return _shard_tensor_native(data, axis_idx, num_shards, local_rank)

    # Fallback: just return the tensor
    return data


def distribute_data(x, y, use_dtensor=None):
    """Distribute input data for model parallelism.
    
    This function ensures that input tensors are properly distributed
    when the model uses sharded weights. Uses native PyTorch sharding
    by default for better compatibility.
    
    Args:
        x: Input data tensor
        y: Target data tensor
        use_dtensor: If True, force DTensor usage. If False, use native sharding.
                    If None (default), use native sharding.
        
    Returns:
        Tuple of (distributed x, distributed y)
    """
    from keras.src.distribution import distribution_lib as frontend_dist
    
    dist = frontend_dist.distribution()
    if dist is None:
        return x, y
    
    # Get the data layout for the input
    if hasattr(x, 'shape'):
        x_layout = dist.get_data_layout(x.shape)
        x = distribute_data_input(x, x_layout, dist.batch_dim_name, use_dtensor=use_dtensor)
    
    if hasattr(y, 'shape'):
        y_layout = dist.get_data_layout(y.shape)
        y = distribute_data_input(y, y_layout, dist.batch_dim_name, use_dtensor=use_dtensor)
    
    return x, y


def num_processes():
    """Return the total number of processes in the cluster."""
    return _get_world_size()


def process_id():
    """Return the rank of the current process."""
    return _get_current_rank()


def device_id():
    """Return the local device ID."""
    if torch.cuda.is_available():
        return torch.cuda.current_device()
    return 0


def backend_num_processes():
    return num_processes()


def gather_sharded_tensor(tensor, mesh_axis: int, num_shards: int) -> torch.Tensor:
    """Gather a sharded tensor back to full size.
    
    This is useful for getting the full tensor after a sharded operation.
    
    Args:
        tensor: The local shard
        mesh_axis: The axis along which tensors were sharded
        num_shards: Number of shards
        
    Returns:
        The gathered full tensor
    """
    return _gather_tensors_native(tensor, mesh_axis, num_shards)


def all_reduce_tensor(tensor, reduce_op: str = "sum") -> torch.Tensor:
    """Perform all-reduce operation on a tensor across all processes.
    
    Args:
        tensor: The tensor to reduce
        reduce_op: Reduction operation ("sum", "product", "min", "max", "avg")
        
    Returns:
        The reduced tensor (same on all processes)
    """
    if not dist.is_initialized():
        return tensor
    
    # Convert string op to torch distributed op
    op_map = {
        "sum": dist.ReduceOp.SUM,
        "product": dist.ReduceOp.PRODUCT,
        "min": dist.ReduceOp.MIN,
        "max": dist.ReduceOp.MAX,
        "avg": dist.ReduceOp.AVG,
    }
    reduce_op_enum = op_map.get(reduce_op.lower(), dist.ReduceOp.SUM)
    
    # Clone to avoid in-place modification issues
    output = tensor.clone()
    dist.all_reduce(output, reduce_op_enum)
    return output

