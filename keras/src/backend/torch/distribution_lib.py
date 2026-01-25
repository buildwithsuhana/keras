import torch
import torch.distributed as dist
import numpy as np
import re

# Use the system's preferred DeviceMesh import path
from torch.distributed.device_mesh import DeviceMesh as TorchDeviceMesh
from torch.distributed._tensor import Replicate, Shard
from torch.distributed._tensor import distribute_tensor as torch_distribute_tensor

from keras.src.backend.torch.core import convert_to_tensor


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


def distribute_data_input(data, layout):
    """Shard input data batches."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    if hasattr(data, "shape"):
        print(f"[Rank {rank}] Sharding input batch. Local shape: {data.shape}")
    return distribute_tensor(data, layout)


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