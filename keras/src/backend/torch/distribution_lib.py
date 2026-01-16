import torch
import numpy as np
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor import distribute_tensor as torch_distribute_tensor

def list_devices(device_type=None):
    """Return available torch devices."""
    if device_type is None:
        device_type = "cuda" if torch.cuda.is_available() else "cpu"

    if device_type == "cuda":
        return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    return ["cpu"]

def get_device_count(device_type=None):
    """Returns the number of available torch devices."""
    return len(list_devices(device_type))

def num_processes():
    """Return the current world size."""
    return (
        torch.distributed.get_world_size()
        if torch.distributed.is_initialized()
        else 1
    )

def process_id():
    """Return the current rank."""
    return (
        torch.distributed.get_rank()
        if torch.distributed.is_initialized()
        else 0
    )

def initialize(job_addresses=None, num_processes=None, process_id=None):
    """Initializes the process group for distributed computation."""
    # Note: In most Keras setups, this is handled via mp.spawn or torchrun.
    if not torch.distributed.is_initialized():
        # Implementation depends on the environment (e.g., NCCL for GPUs)
        pass

def _to_backend_mesh(device_mesh):
    """Convert Keras DeviceMesh to PyTorch DeviceMesh."""
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    # init_device_mesh coordinates the physical hardware into the logical grid
    return init_device_mesh(
        device_type=device_type,
        mesh_shape=device_mesh.shape,
        mesh_dim_names=device_mesh.axis_names
    )

def _to_backend_layout(tensor_layout):
    """Convert Keras TensorLayout to Torch Placement types."""
    if tensor_layout.device_mesh is None:
        raise ValueError("Device mesh must be set for TensorLayout.")

    mesh_axis_names = tensor_layout.device_mesh.axis_names
    # 1. Initialize all placements to Replicate for every dimension of the MESH.
    # PyTorch DTensor requires exactly one placement per mesh dimension.
    placements = [Replicate()] * len(mesh_axis_names)

    # 2. Map tensor dimensions to the correct mesh axes.
    # We iterate over the TENSOR dimensions (i) and find their logical axis name.
    for i, axis in enumerate(tensor_layout.axes):
        if axis is not None:
            if axis not in mesh_axis_names:
                raise ValueError(f"Axis {axis} not found in {mesh_axis_names}")
            
            # 3. Find the index of this logical axis in the physical DEVICE MESH.
            mesh_dim_index = mesh_axis_names.index(axis)
            
            # 4. Correct Logic: Shard the i-th dimension of the TENSOR 
            # over the mesh_dim_index-th dimension of the MESH.
            placements[mesh_dim_index] = Shard(i)
            
    return placements

def distribute_tensor(tensor, layout):
    """Shard/replicate a tensor across the mesh using PyTorch DTensor."""
    from keras.src.distribution import TensorLayout
    if isinstance(layout, TensorLayout):
        # Physical sharding via PyTorch's native DTensor API
        print(f"Distributing tensor with shape {tensor.shape} using placements {layout.backend_layout}")
        return torch_distribute_tensor(
            tensor, 
            layout.device_mesh.backend_mesh, 
            layout.backend_layout
        )
    return tensor