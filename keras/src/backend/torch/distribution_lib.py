"""Distribution related utilities for the PyTorch backend."""

import torch
import numpy as np
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._tensor import DeviceMesh as TorchDeviceMesh
from torch.distributed._tensor import Shard, Replicate, distribute_tensor as torch_distribute_tensor

def list_devices(device_type=None):
    """Return all available global devices."""
    if device_type == "cpu":
        return ["cpu:0"]
    
    num_gpus = torch.cuda.device_count()
    if device_type in ["gpu", "cuda"] or device_type is None:
        if num_gpus > 0:
            return [f"cuda:{i}" for i in range(num_gpus)]
    
    return ["cpu:0"]

def get_device_count(device_type=None):
    """Returns the number of available devices."""
    if device_type == "cpu":
        return 1
    if device_type in ["gpu", "cuda"] or device_type is None:
        return torch.cuda.device_count()
    return 0

def num_processes():
    """Return the number of processes in the distributed group."""
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1

def process_id():
    """Return the current process ID."""
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0

def initialize(job_addresses=None, num_processes=None, process_id=None):
    """Initialize the torch distributed backend."""
    if not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend)

def distribute_variable(value, layout):
    """Distribute a Keras Variable (weight) using DTensor."""
    return distribute_tensor(value, layout)

def distribute_tensor(tensor, layout):
    """Convert a torch.Tensor to a DTensor based on the layout."""
    from keras.src.distribution import TensorLayout

    if isinstance(layout, TensorLayout):
        torch_mesh = layout.device_mesh.backend_mesh
        torch_placements = _to_backend_layout(layout)
        
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(tensor)
            
        device = torch_mesh.device_type
        tensor = tensor.to(device)
        
        return torch_distribute_tensor(tensor, torch_mesh, torch_placements)
    return tensor

def _to_backend_mesh(device_mesh):
    """Convert Keras DeviceMesh to torch.distributed.DeviceMesh."""
    mesh_shape = device_mesh.shape
    axis_names = device_mesh.axis_names
    
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    ranks = np.arange(np.prod(mesh_shape)).reshape(mesh_shape)
    
    return init_device_mesh(device_type, mesh_shape, mesh_dim_names=axis_names)

def _to_backend_layout(tensor_layout):
    """Convert Keras TensorLayout axes to torch Placement objects."""
    placements = []
    mesh = tensor_layout.device_mesh
    
    for axis_name in tensor_layout.axes:
        if axis_name is None:
            placements.append(Replicate())
        else:
            dim_index = mesh.axis_names.index(axis_name)
            placements.append(Shard(dim_index))
            
    return placements