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
    """Bridge for DeviceMesh.backend_mesh"""
    # If devices is None or empty, let Torch handle it automatically
    device_ids = None
    if device_mesh.devices:
        # Simple conversion if strings like ['cuda:0'] were passed
        import re
        device_ids = []
        for d in device_mesh.devices:
            if isinstance(d, str):
                res = re.search(r'\d+', d)
                device_ids.append(int(res.group()) if res else 0)
            else:
                device_ids.append(d)

    return TorchDeviceMesh(
        device_type="cuda" if torch.cuda.is_available() else "cpu",
        mesh_shape=device_mesh.shape,
        mesh_dim_names=device_mesh.axis_names,
        device_ids=device_ids
    )

def _to_backend_layout(layout):
    torch_mesh = layout.device_mesh.backend_mesh
    placements = []
    for axis in layout.axes:
        if axis is None:
            placements.append(Replicate())
        else:
            dim_index = torch_mesh.mesh_dim_names.index(axis)
            placements.append(Shard(dim_index))
    return placements


def distribute_variable(value, layout):
    rank = dist.get_rank() if dist.is_initialized() else 0
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

    # layout.backend_layout and layout.device_mesh.backend_mesh 
    # trigger the _to_backend hooks automatically.
    placements = layout.backend_layout
    torch_mesh = layout.device_mesh.backend_mesh

    if not isinstance(value, torch.Tensor):
        value = convert_to_tensor(value)

    # Wrap the tensor in the PyTorch DTensor dispatcher
    return torch_distribute_tensor(value, torch_mesh, placements)


def distribute_data_input(data, layout):
    rank = dist.get_rank() if dist.is_initialized() else 0
    # Log data sharding. This confirms EpochIterator is working.
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