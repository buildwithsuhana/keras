import torch
import torch.distributed as dist
from torch.distributed.tensor import DeviceMesh as TorchDeviceMesh
from torch.distributed.tensor import Replicate
from torch.distributed.tensor import Shard
from torch.distributed.tensor import (
    distribute_tensor as torch_distribute_tensor,
)

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
    """Bridge for DeviceMesh.backend_mesh.
    Called as: _to_backend_mesh(device_mesh_instance)
    """
    if dist.is_initialized() and dist.get_rank() == 0:
        print(f"[BACKEND] Initializing TorchDeviceMesh with shape {device_mesh.shape}")
    
    return TorchDeviceMesh(
        device_type="cuda" if torch.cuda.is_available() else "cpu",
        mesh_shape=device_mesh.shape,
        mesh_dim_names=device_mesh.axis_names,
    )

def _to_backend_layout(layout):
    """Bridge for TensorLayout.backend_layout.
    Called as: _to_backend_layout(layout_instance)
    """
    # 1. Get the native torch mesh from the Keras DeviceMesh
    # layout.device_mesh is a Keras DeviceMesh; .backend_mesh is the TorchDeviceMesh
    torch_mesh = layout.device_mesh.backend_mesh
    
    # 2. Map Keras axes to PyTorch Placements
    placements = []
    for axis in layout.axes:
        if axis is None:
            placements.append(Replicate())
        else:
            # Look up dimension index from the named mesh axes
            try:
                dim_index = torch_mesh.mesh_dim_names.index(axis)
                placements.append(Shard(dim_index))
            except ValueError:
                raise ValueError(
                    f"Invalid axis name '{axis}'. Valid axes: {torch_mesh.mesh_dim_names}"
                )
    return placements


def distribute_variable(value, layout):
    rank = dist.get_rank()
    # Log weight sharding specifically
    if rank == 0:
        print(f"[BACKEND] Sharding variable. Placements: {layout.axes}")
    return distribute_tensor(value, layout)


def distribute_tensor(value, layout):
    """The core engine for Model Parallelism.

    Converts a standard torch.Tensor into a sharded DTensor.
    """
    from keras.src.distribution.distribution_lib import TensorLayout

    if not isinstance(layout, TensorLayout):
        return value

    # layout.backend_layout contains the Placements list from to_backend_layout
    placements = layout.backend_layout
    torch_mesh = layout.device_mesh.to_backend_mesh()

    if not isinstance(value, torch.Tensor):
        value = convert_to_tensor(value)

    # Wrap the tensor in the PyTorch DTensor dispatcher
    return torch_distribute_tensor(value, torch_mesh, placements)


def distribute_data_input(data, layout):
    rank = dist.get_rank()
    # Log data sharding. This confirms EpochIterator is working.
    # We show the local shape to prove the batch was split.
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

# These are also often expected by various DataAdapter logic
def device_id():
    """Return the local device ID."""
    if torch.cuda.is_available():
        return torch.cuda.current_device()
    return 0

def backend_num_processes():
    return num_processes()