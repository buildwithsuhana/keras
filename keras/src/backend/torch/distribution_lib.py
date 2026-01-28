import os
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor as torch_distribute_tensor
from torch.distributed.tensor import Shard, Replicate

def list_devices(device_type=None):
    """Return all available devices based on the device type."""
    device_type = device_type.lower() if device_type else None
    if torch.cuda.is_available():
        return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    elif hasattr(torch, "mps") and torch.mps.is_available():
        return ["mps:0"]
    return ["cpu:0"]

def get_device_count(device_type=None):
    """Return the number of available devices."""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 1

def _to_backend_mesh(device_mesh):
    """Convert Keras DeviceMesh to PyTorch DeviceMesh."""
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    return init_device_mesh(
        device_type, 
        device_mesh.shape, 
        mesh_dim_names=device_mesh.axis_names
    )

def _to_backend_layout(tensor_layout):
    """Convert Keras TensorLayout to PyTorch Placements."""
    if tensor_layout.device_mesh is None:
        raise ValueError("Cannot create sharding without a device mesh.")
    
    mesh = tensor_layout.device_mesh.backend_mesh
    placements = []
    
    for axis in tensor_layout.axes:
        if axis is None:
            placements.append(Replicate())
        else:
            dim = tensor_layout.device_mesh.axis_names.index(axis)
            placements.append(Shard(dim))
            
    return {"mesh": mesh, "placements": placements}

def distribute_tensor(tensor, layout):
    """Distribute a tensor using DTensor."""
    if layout is None:
        return tensor

    backend_layout = _to_backend_layout(layout) if hasattr(layout, "axes") else layout
    
    return torch_distribute_tensor(
        tensor, 
        device_mesh=backend_layout["mesh"], 
        placements=backend_layout["placements"]
    )

def distribute_variable(value, layout):
    """Create a distributed variable (DTensor) for Torch."""
    return distribute_tensor(value, layout)

def distribute_data_input(per_process_batch, layout, batch_dim_name):
    """Distribute input data using DTensor."""
    return distribute_tensor(per_process_batch, layout)

def initialize(job_addresses=None, num_processes=None, process_id=None):
    """Initialize the distributed backend."""
    if num_processes is None or num_processes <= 1:
        return

    if job_addresses:
        coordinator_address = job_addresses.split(",")[0]
        os.environ["MASTER_ADDR"] = coordinator_address.split(":")[0]
        os.environ["MASTER_PORT"] = "29500"

    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(
            backend=backend,
            rank=process_id if process_id is not None else 0,
            world_size=num_processes,
        )

def num_processes():
    return dist.get_world_size() if dist.is_initialized() else 1

def process_id():
    return dist.get_rank() if dist.is_initialized() else 0

def all_gather_variable(variable):
    """Convert a DTensor back to a local replicated tensor."""
    if hasattr(variable, "full_tensor"):
        return variable.full_tensor()
    return variable