"""
This module implements the backend-facing functions used by
`keras.src.distribution.distribution_lib` for the PyTorch backend.
"""

import functools
import os

import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor import Replicate
from torch.distributed.tensor import Shard
from torch.distributed.tensor import distribute_tensor as torch_distribute_tensor


def list_devices(device_type=None):
    """Return all the available devices based on the device type."""
    device_type = (device_type or "gpu").lower()
    if torch.distributed.is_initialized():
        count = torch.distributed.get_world_size()
    elif "WORLD_SIZE" in os.environ:
        count = int(os.environ["WORLD_SIZE"])
    else:
        # Fallback to local device count
        if device_type.startswith("gpu"):
            count = torch.cuda.device_count() or 1
        else:
            count = 1
    return [f"{device_type}:{i}" for i in range(count)]


def get_device_count(device_type=None):
    """Returns the number of available devices."""
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    device_type = (device_type or "gpu").lower()
    if device_type.startswith("gpu"):
        return torch.cuda.device_count() or 1
    return 1


def initialize(job_addresses=None, num_processes=None, process_id=None):
    """Initialize the process group for distributed training."""
    if torch.distributed.is_initialized():
        return

    if job_addresses:
        master_addr, master_port = job_addresses.split(",")[0].split(":")
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port

    if num_processes is not None:
        os.environ["WORLD_SIZE"] = str(num_processes)

    if process_id is not None:
        os.environ["RANK"] = str(process_id)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    torch.distributed.init_process_group(backend=backend)


def num_processes():
    """Return the number of processes in the current process group."""
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    return 1


def process_id():
    """Return the rank of the current process."""
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    if "RANK" in os.environ:
        return int(os.environ["RANK"])
    return 0


def _to_backend_mesh(keras_mesh):
    """Convert Keras DeviceMesh to PyTorch DeviceMesh."""
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    return init_device_mesh(
        device_type,
        mesh_shape=tuple(keras_mesh.shape),
        mesh_dim_names=tuple(keras_mesh.axis_names),
    )


def _to_backend_layout(tensor_layout):
    """Convert Keras TensorLayout to PyTorch DTensor layout (mesh, placements)."""
    if tensor_layout is None:
        return None

    keras_mesh = tensor_layout.device_mesh
    if keras_mesh.backend_mesh is not None:
        torch_mesh = keras_mesh.backend_mesh
    else:
        torch_mesh = _to_backend_mesh(keras_mesh)

    placements = []
    for mesh_dim_name in keras_mesh.axis_names:
        shard_dim = None
        if tensor_layout.axes is not None:
            for tensor_dim, axis_name in enumerate(tensor_layout.axes):
                if axis_name == mesh_dim_name:
                    shard_dim = tensor_dim
                    break
        if shard_dim is not None:
            placements.append(Shard(shard_dim))
        else:
            placements.append(Replicate())

    return DTensorLayout(torch_mesh, tuple(placements))


class DTensorLayout:
    """Wraps a torch DeviceMesh + placements for use as a backend layout."""

    def __init__(self, device_mesh, placements):
        self.device_mesh = device_mesh
        self.placements = placements


def distribute_tensor(tensor, layout):
    """Distribute a tensor according to the layout."""
    if layout is None:
        return tensor

    from keras.src.distribution.distribution_lib import TensorLayout

    if isinstance(layout, TensorLayout):
        layout = layout.backend_layout

    return torch_distribute_tensor(
        tensor, device_mesh=layout.device_mesh, placements=layout.placements
    )


def distribute_variable(value, layout):
    """Distribute a variable according to the layout."""
    dtensor = distribute_tensor(value, layout)
    if isinstance(value, torch.nn.Parameter):
        return torch.nn.Parameter(dtensor, requires_grad=value.requires_grad)
    return dtensor


def distribute_data_input(per_process_batch, layout, batch_dim_name):
    """Distribute the input data according to the layout."""
    if layout is None or isinstance(per_process_batch, DTensor):
        return per_process_batch

    from keras.src.backend.common import global_state

    dist = global_state.get_global_attribute("distribution")
    if dist is None or dist.__class__.__name__ != "ModelParallel":
        return per_process_batch

    from keras.src.distribution.distribution_lib import TensorLayout

    if isinstance(layout, TensorLayout):
        layout = layout.backend_layout

    if not isinstance(layout, DTensorLayout):
        return per_process_batch

    return DTensor.from_local(
        per_process_batch,
        device_mesh=layout.device_mesh,
        placements=layout.placements,
    )


def maybe_distribute_tensor(tensor):
    """Distribute a tensor if ModelParallel is active."""
    if (
        not isinstance(tensor, torch.Tensor)
        or isinstance(tensor, DTensor)
        or tensor.device.type == "meta"
        or not torch.distributed.is_available()
        or not torch.distributed.is_initialized()
    ):
        return tensor

    from keras.src.backend.common import global_state

    dist = global_state.get_global_attribute("distribution")
    if dist is not None and dist.__class__.__name__ == "ModelParallel":
        from keras.src.distribution.distribution_lib import TensorLayout

        return distribute_tensor(
            tensor, TensorLayout([None] * tensor.ndim, dist.device_mesh)
        )
    return tensor


def distribute_output(fn):
    """Decorator to ensure that the output of an op is distributed.

    This should be used for factory and transformation ops that don't
    naturally propagate distribution through DTensor.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        res = fn(*args, **kwargs)
        from keras.src.backend.torch import core as torch_core

        return torch_core.convert_to_tensor(res)

    return wrapper
