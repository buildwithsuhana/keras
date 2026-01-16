import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate
from torch.distributed.tensor import Shard
from torch.distributed.tensor import (
    distribute_tensor as torch_distribute_tensor,
)


def list_devices(device_type=None):
    """Return available torch devices."""
    if device_type is None:
        device_type = "cuda" if torch.cuda.is_available() else "cpu"

    if device_type == "cuda":
        return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    return ["cpu"]


def num_processes():
    return (
        torch.distributed.get_world_size()
        if torch.distributed.is_initialized()
        else 1
    )


def process_id():
    return (
        torch.distributed.get_rank()
        if torch.distributed.is_initialized()
        else 0
    )


def _to_backend_mesh(device_mesh):
    """Convert Keras DeviceMesh to PyTorch DeviceMesh."""
    # init_device_mesh handles the mapping of ranks to the logical grid
    return init_device_mesh(
        device_type=device_mesh.devices.flatten()[0].split(":")[0],
        mesh_shape=device_mesh.shape,
        mesh_dim_names=device_mesh.axis_names,
    )


def _to_backend_layout(tensor_layout):
    """Convert Keras TensorLayout to Torch Placement types."""
    if tensor_layout.device_mesh is None:
        raise ValueError("Device mesh must be set for TensorLayout.")

    placements = []
    for axis in tensor_layout.axes:
        if axis is None:
            placements.append(Replicate())
        else:
            # Map the axis name to the dimension index in the mesh
            dim_index = tensor_layout.device_mesh.axis_names.index(axis)
            placements.append(Shard(dim_index))
    return placements


def distribute_tensor(tensor, layout):
    """Shard a tensor across the mesh using DTensor."""
    from keras.src.distribution import TensorLayout

    if isinstance(layout, TensorLayout):
        torch_mesh = layout.device_mesh.backend_mesh
        torch_placements = layout.backend_layout
        return torch_distribute_tensor(tensor, torch_mesh, torch_placements)
    return tensor
