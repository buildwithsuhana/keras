"""Utilities for distribution strategy with PyTorch backend."""

import os

import torch
import torch.distributed as dist

try:
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor
    from torch.distributed.tensor import Replicate
    from torch.distributed.tensor import Shard
except ImportError:
    init_device_mesh = None
    DTensor = None
    Replicate = None
    Shard = None


def list_devices(device_type=None):
    """Return all the available devices based on the device type.

    Note that this should return the global devices in a distributed setting.

    Args:
        device_type: string of `"cpu"` or `"gpu"`. Defaults to `"gpu"`
            if available when device_type is not provided. Otherwise
            will return the `"cpu"` devices.

    Return:
        List of devices that are available for distribute computation.
    """
    device_type = device_type or "gpu"
    if dist.is_initialized():
        count = dist.get_world_size()
    elif "WORLD_SIZE" in os.environ:
        count = int(os.environ["WORLD_SIZE"])
    else:
        if device_type.lower() == "gpu":
            count = torch.cuda.device_count() or 1
        else:
            count = 1
    return [f"{device_type.lower()}:{i}" for i in range(count)]


def get_device_count(device_type=None):
    """Returns the number of available PyTorch devices.

    Args:
        device_type: Optional device type to count (e.g., "cpu", "gpu").
            If `None`, it defaults to counting "gpu" devices if
            available, otherwise it counts "cpu" devices.
    Returns:
        int: The total number of PyTorch devices for the specified type.
    """
    if dist.is_initialized():
        return dist.get_world_size()
    elif "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    else:
        device_type = device_type or "gpu"
        if device_type.lower() == "gpu":
            return torch.cuda.device_count() or 1
        else:
            return 1


def initialize(job_addresses=None, num_processes=None, process_id=None):
    """Initializes the global distributed environment."""
    if dist.is_initialized():
        return

    # In PyTorch, we mostly rely on environment variables set by torchrun.
    # But if they are not set, we might need to set some defaults for
    # single-node testing if needed, though usually torchrun is preferred.
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    
    # LOCAL_RANK is set by torchrun
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    dist.init_process_group(backend=backend)


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


def _to_backend_mesh(keras_mesh):
    """Convert Keras DeviceMesh to PyTorch DeviceMesh."""
    if init_device_mesh is None:
        raise ImportError(
            "PyTorch distributed components (DeviceMesh, DTensor) are "
            "not available. Please upgrade to torch >= 2.2.0."
        )
    # PyTorch uses "cuda" for GPU mesh.
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    return init_device_mesh(
        device_type,
        mesh_shape=tuple(keras_mesh.shape),
        mesh_dim_names=tuple(keras_mesh.axis_names),
    )


def _to_backend_layout(tensor_layout):
    """Convert Keras TensorLayout to PyTorch (DeviceMesh, placements)."""
    if Shard is None:
        raise ImportError(
            "PyTorch distributed components (DTensor, Shard, Replicate) are "
            "not available. Please upgrade to torch >= 2.2.0."
        )

    keras_mesh = tensor_layout.device_mesh
    torch_mesh = keras_mesh.backend_mesh

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

    return torch_mesh, tuple(placements)


def distribute_tensor(tensor, layout):
    """Distribute the tensor based on the layout."""
    if DTensor is None:
        raise ImportError(
            "PyTorch DTensor is not available. Please upgrade to torch >= 2.2.0."
        )
    from keras.src.distribution import TensorLayout

    if isinstance(layout, TensorLayout):
        torch_mesh, placements = _to_backend_layout(layout)
    else:
        torch_mesh, placements = layout

    import torch.distributed.tensor as dist_tensor
    return dist_tensor.distribute_tensor(
        tensor, device_mesh=torch_mesh, placements=placements
    )


def distribute_variable(value, layout):
    """Distribute the variable based on the layout."""
    dtensor = distribute_tensor(value, layout)
    return torch.nn.Parameter(dtensor, requires_grad=True)


def distribute_data_input(per_process_batch, layout, batch_dim_name=None):
    """Distribute the input data with the corresponding layout."""
    if layout is None:
        return per_process_batch

    if DTensor is None:
        raise ImportError(
            "PyTorch DTensor is not available. Please upgrade to torch >= 2.2.0."
        )

    from keras.src.distribution import TensorLayout

    if isinstance(layout, TensorLayout):
        torch_mesh, placements = _to_backend_layout(layout)
    else:
        torch_mesh, placements = layout

    if not isinstance(per_process_batch, torch.Tensor):
        from keras.src.backend.torch.core import convert_to_tensor
        per_process_batch = convert_to_tensor(per_process_batch)

    # Ensure it is on the mesh device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if per_process_batch.device.type != device:
        per_process_batch = per_process_batch.to(device)

    return DTensor.from_local(
        per_process_batch, device_mesh=torch_mesh, placements=placements
    )


def all_reduce(tensor, op="sum"):
    """Perform all-reduce on the tensor."""
    if not dist.is_initialized():
        return tensor

    if DTensor is not None and isinstance(tensor, DTensor):
        # DTensors are already distributed, we don't all_reduce them
        # in the same way. Typically we'd gather them if needed, 
        # but for metrics sync we mostly care about regular tensors.
        return tensor

    if op.lower() == "sum":
        reduce_op = dist.ReduceOp.SUM
    elif op.lower() == "mean":
        reduce_op = dist.ReduceOp.SUM # We'll divide by world size after
    else:
        raise ValueError(f"Unsupported reduce op: {op}")

    dist.all_reduce(tensor, op=reduce_op)
    
    if op.lower() == "mean":
        tensor /= dist.get_world_size()
    
    return tensor
