"""Utilities for distribution strategy with Torch backend."""

import torch
import torch.distributed as dist
import numpy as np
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor import distribute_tensor as distribute_tensor_torch

from keras.src.backend.torch.core import get_device
from keras.src.backend.torch.core import to_torch_dtype


def list_devices(device_type=None):
    """Return all the available devices based on the device type.

    Note that this should return the global devices in a distributed setting.

    Args:
        device_type: string of `"cpu"`, `"gpu"` or `"tpu"`. Defaults to `"gpu"`
            or `"tpu"` if available when device_type is not provided. Otherwise
            will return the `"cpu"` devices.

    Return:
        List of devices that are available for distribute computation.
    """
    if device_type is None:
        if torch.cuda.is_available():
            device_type = "cuda"
        elif torch.backends.mps.is_available():
            device_type = "mps"
        else:
            device_type = "cpu"
    
    device_type = device_type.lower()
    if "gpu" in device_type:
        device_type = "cuda"

    if device_type == "cuda":
        count = torch.cuda.device_count()
    elif device_type == "mps":
        count = 1 if torch.backends.mps.is_available() else 0
    else:
        count = 1 # Default for CPU

    if dist.is_initialized():
        world_size = dist.get_world_size()
        # In a distributed setting, we return all global devices.
        # Assuming one device per rank for simplicity here.
        return [f"{device_type}:{i}" for i in range(world_size)]
    
    return [f"{device_type}:{i}" for i in range(count)]


def get_device_count(device_type=None):
    """Returns the number of available devices.
    Args:
        device_type: Optional device type to count (e.g., "cpu", "gpu", "tpu").
            If `None`, it defaults to counting "gpu" or "tpu" devices if
            available, otherwise it counts "cpu" devices.
    Returns:
        int: The total number of devices for the specified type.
    """
    return len(list_devices(device_type))


def distribute_variable(value, layout):
    """Create a distributed variable for Torch."""
    return distribute_tensor(value, layout)


def distribute_tensor(tensor, layout):
    """Distribute the tensor based on the layout."""
    # Avoid circular imports.
    from keras.src.distribution import TensorLayout

    if isinstance(layout, TensorLayout):
        torch_mesh = layout.device_mesh.backend_mesh
        placements = _get_placements(layout)
        
        if isinstance(tensor, DTensor):
            if tensor.device_mesh == torch_mesh and tensor.placements == tuple(placements):
                return tensor
            return tensor.redistribute(torch_mesh, placements)
        
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(tensor, device=get_device())
        
        # Ensure tensor is on the correct device for the mesh
        if tensor.device.type != torch_mesh.device_type:
            tensor = tensor.to(torch_mesh.device_type)

        return distribute_tensor_torch(tensor, torch_mesh, placements)

    return tensor


def distribute_data_input(per_process_batch, layout, batch_dim_name):
    """Distribute the input data with the corresponding layout."""
    from keras.src.distribution import TensorLayout

    if isinstance(layout, TensorLayout):
        torch_mesh = layout.device_mesh.backend_mesh
        placements = _get_placements(layout)
        
        if not isinstance(per_process_batch, torch.Tensor):
            per_process_batch = torch.as_tensor(per_process_batch, device=get_device())
            
        if per_process_batch.device.type != torch_mesh.device_type:
            per_process_batch = per_process_batch.to(torch_mesh.device_type)

        return DTensor.from_local(per_process_batch, torch_mesh, placements)
    
    return per_process_batch


def initialize_rng():
    """Initializes the global random number generator across processes."""
    from keras.src.utils import rng_utils
    global_seed = rng_utils.get_random_seed()
    if global_seed is None:
        if dist.is_initialized():
            if dist.get_rank() == 0:
                seed = np.random.randint(0, 2**31)
            else:
                seed = 0
            
            # Match tensor device to backend
            device = "cuda" if dist.get_backend() == "nccl" else "cpu"
            seed_tensor = torch.tensor([seed], dtype=torch.int64, device=device)
            dist.broadcast(seed_tensor, src=0)
            global_seed = int(seed_tensor.item())
            rng_utils.set_random_seed(global_seed)


def initialize(job_addresses, num_processes, process_id):
    if dist.is_initialized():
        return

    import os
    if job_addresses:
        coordinator_address = job_addresses.split(",")[0]
        if ":" in coordinator_address:
            addr, port = coordinator_address.split(":")
        else:
            addr, port = coordinator_address, "12345"
        os.environ["MASTER_ADDR"] = addr
        os.environ["MASTER_PORT"] = port
    
    if num_processes is not None:
        os.environ["WORLD_SIZE"] = str(num_processes)
    if process_id is not None:
        os.environ["RANK"] = str(process_id)
        
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if backend == "nccl":
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

    dist.init_process_group(backend=backend)
    
    initialize_rng()


def num_processes():
    """Return the number of processes for the current distribution setting."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def process_id():
    """Return the current process ID for the distribution setting."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def _to_backend_mesh(device_mesh):
    """Convert the DeviceMesh to Torch backend specific Mesh."""
    mesh_shape = device_mesh.devices.shape
    mesh_dim_names = device_mesh.axis_names
    
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    
    # In Torch, init_device_mesh handles process group creation if needed.
    return init_device_mesh(device_type, mesh_shape, mesh_dim_names=mesh_dim_names)


def _to_backend_layout(tensor_layout):
    """Convert the TensorLayout to Torch backend specific Placements."""
    return _get_placements(tensor_layout)


def _maybe_distribute_input(x, distribution):
    """Distribute the input data if it's not already a DTensor."""
    if isinstance(x, torch.Tensor) and not isinstance(x, DTensor):
        layout = distribution.get_data_layout(x.shape)
        return distribute_tensor(x, layout)
    return x


def _get_placements(layout):
    mesh = layout.device_mesh
    axes = layout.axes
    placements = []
    for mesh_axis_name in mesh.axis_names:
        found = False
        for i, axis_name in enumerate(axes):
            if axis_name == mesh_axis_name:
                placements.append(Shard(i))
                found = True
                break
        if not found:
            placements.append(Replicate())
    return placements
