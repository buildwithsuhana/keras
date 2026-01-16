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
    """Shards a tensor across the DeviceMesh using PyTorch DTensor."""
    from keras.src.distribution import TensorLayout
    
    # If it's already a DTensor or sharded, return as is
    if hasattr(tensor, "placements"):
        return tensor

    if isinstance(layout, TensorLayout) and layout.device_mesh is not None:
        torch_mesh = layout.device_mesh.backend_mesh
        torch_placements = _to_backend_layout(layout)
        
        # 1. Convert to torch.Tensor if it's numpy/list
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(tensor)
            
        # 2. Fix device mismatch: Move to CPU before sharding if it's on the wrong GPU
        # This is critical for Rank 1 to avoid seeing data on cuda:0
        if tensor.device.type == "cuda" and tensor.get_device() != torch.cuda.current_device():
            tensor = tensor.cpu()
        
        # 3. Move to current local device
        tensor = tensor.to(torch_mesh.device_type)
        
        # 4. Shard the tensor
        # We disable run_check for performance, but it can be True for debugging
        return torch_distribute_tensor(
            tensor, 
            torch_mesh, 
            torch_placements,
            run_check=False
        )
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