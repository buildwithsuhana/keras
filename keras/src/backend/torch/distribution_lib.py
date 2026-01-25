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


def to_backend_mesh(shape, axis_names):
    """Maps to frontend DeviceMesh.to_backend_mesh().

    Returns a native PyTorch DeviceMesh object.
    """
    return TorchDeviceMesh(
        device_type="cuda" if torch.cuda.is_available() else "cpu",
        mesh_shape=shape,
        mesh_dim_names=axis_names,
    )


def to_backend_layout(axes, device_mesh):
    """Maps to frontend TensorLayout.backend_layout.

    Translates Keras axis names into a list of PyTorch Placements.
    """
    # Get the native torch.distributed.DeviceMesh
    torch_mesh = device_mesh.to_backend_mesh()

    placements = []
    for axis in axes:
        if axis is None:
            placements.append(Replicate())
        else:
            # Look up the integer index of the named axis within the mesh
            try:
                dim_index = torch_mesh.mesh_dim_names.index(axis)
                placements.append(Shard(dim_index))
            except ValueError:
                raise ValueError(
                    f"Invalid axis name '{axis}'. Must be one of "
                    f"{torch_mesh.mesh_dim_names}"
                )
    return placements


def distribute_variable(value, layout):
    """Internal hook used during KerasVariable initialization."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        print(f"[DIST] Sharding variable with layout: {layout.axes}")
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


def distribute_data_input(per_process_batch, layout, batch_dim_name):
    rank = dist.get_rank() if dist.is_initialized() else 0
    local_shape = per_process_batch.shape
    
    print(f"[Rank {rank}] Distributing input batch. Local shape: {local_shape}")
    
    mesh = layout.device_mesh.to_backend_mesh()
    return torch_distribute_tensor(per_process_batch, mesh, [Shard(0)])

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