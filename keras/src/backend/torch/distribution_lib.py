import torch
import torch.distributed as dist
import numpy as np
import re

# Use the system's preferred DeviceMesh import path
from torch.distributed.device_mesh import DeviceMesh as TorchDeviceMesh
from torch.distributed._tensor import DTensor, Replicate, Shard
from torch.distributed._tensor import distribute_tensor as torch_distribute_tensor

from keras.src.backend.torch.core import convert_to_tensor


def _is_dtensor(tensor):
    """Check if a tensor is a PyTorch DTensor."""
    return isinstance(tensor, DTensor)


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
        local_rank = dist.get_rank() % torch.cuda.device_count()
        torch.cuda.set_device(local_rank)


def _to_backend_mesh(device_mesh):
    """Bridge for DeviceMesh.backend_mesh"""
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


def distribute_variable(variable, layout):
    """Intercept variable creation to shard it immediately.

    Args:
        variable: Either a Keras Variable object or a torch.Tensor
        layout: The TensorLayout specifying how to shard the variable

    Returns:
        A torch.nn.Parameter containing the sharded DTensor
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        print(f"[BACKEND] Sharding variable. Placements: {layout.axes}")

    # Handle both Keras Variable and torch.Tensor inputs
    if hasattr(variable, 'value'):
        # It's a Keras Variable - get the underlying tensor
        tensor_value = variable.value
    else:
        # It's already a torch.Tensor
        tensor_value = variable

    # Distribute the tensor (convert to DTensor)
    sharded_tensor = distribute_tensor(tensor_value, layout)

    # Return as Parameter for optimizer compatibility
    return torch.nn.Parameter(sharded_tensor)


def distribute_tensor(value, layout):
    """The core engine for Model Parallelism.
    Converts a standard torch.Tensor into a sharded DTensor.
    """
    from keras.src.distribution.distribution_lib import TensorLayout

    if not isinstance(layout, TensorLayout):
        return value

    # layout.backend_layout and layout.device_mesh.backend_mesh 
    # trigger the _to_backend hooks automatically.
    placements = layout.backend_layout
    torch_mesh = layout.device_mesh.backend_mesh

    if not isinstance(value, torch.Tensor):
        value = convert_to_tensor(value)

    # Wrap the tensor in the PyTorch DTensor dispatcher
    return torch_distribute_tensor(value, torch_mesh, placements)


def distribute_data_input(data, layout, batch_dim_name=None):
    """Shard input data batches.

    Args:
        data: Input data tensor (torch.Tensor or numpy array)
        layout: The TensorLayout specifying how to distribute the data
        batch_dim_name: Optional name of the batch dimension for data parallelism

    Returns:
        A DTensor representation of the input data, properly distributed
        across devices.
    """
    from keras.src.distribution.distribution_lib import TensorLayout

    rank = dist.get_rank() if dist.is_initialized() else 0

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
        # Get the torch mesh and placements
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
            print(f"[Rank {rank}] Distributing input batch. Local shape: {data.shape}, "
                  f"placements: {placements}")

        # Distribute the tensor (convert to DTensor)
        return torch_distribute_tensor(data, torch_mesh, placements)

    # Fallback: just return the tensor
    return data


def num_processes():
    """Return the total number of processes in the cluster."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def process_id():
    """Return the rank of the current process."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def device_id():
    """Return the local device ID."""
    if torch.cuda.is_available():
        return torch.cuda.current_device()
    return 0


def backend_num_processes():
    return num_processes()