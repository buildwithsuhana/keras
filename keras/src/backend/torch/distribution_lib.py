import os

import torch
import torch.distributed as dist


def list_devices(device_type=None):
    device_type = device_type.lower() if device_type else None

    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        return [f"cuda:{i}" for i in range(num_devices)]
    elif hasattr(torch, "mps") and torch.mps.is_available():
        return ["mps:0"]

    if device_type in ("gpu", "cuda"):
        raise RuntimeError("No CUDA devices available")

    if device_type == "cpu":
        return ["cpu:0"]

    raise ValueError(f"Unsupported device_type: {device_type}")


def get_device_count(device_type=None):
    device_type = device_type.lower() if device_type else None

    if torch.cuda.is_available():
        return torch.cuda.device_count()
    elif hasattr(torch, "mps") and torch.mps.is_available():
        return 1

    if device_type in ("gpu", "cuda"):
        return 0

    if device_type == "cpu":
        return 1

    return 0


def distribute_tensor(tensor, layout):
    from keras.src.distribution import TensorLayout

    if isinstance(layout, TensorLayout):
        backend_layout = layout.backend_layout if hasattr(layout, 'backend_layout') else layout
    else:
        backend_layout = layout

    if backend_layout is None:
        return tensor

    if hasattr(backend_layout, 'axes'):
        axes = backend_layout.axes
    else:
        axes = backend_layout

    device_mesh = None
    if hasattr(backend_layout, 'device_mesh') and backend_layout.device_mesh is not None:
        device_mesh = backend_layout.device_mesh

    sharding_axes = [ax for ax in axes if ax is not None]
    if not sharding_axes:
        return tensor

    first_sharding_axis = sharding_axes[0]

    if device_mesh is not None:
        axis_names = device_mesh.axis_names
        if first_sharding_axis in axis_names:
            axis_idx = axis_names.index(first_sharding_axis)
            mesh_dim = device_mesh.shape[axis_idx]

            if tensor.shape[axis_idx] >= mesh_dim:
                if tensor.shape[axis_idx] % mesh_dim == 0:
                    shard_size = tensor.shape[axis_idx] // mesh_dim
                    result = list(torch.split(tensor, shard_size, dim=axis_idx))
                    return result
                else:
                    result = list(torch.chunk(tensor, mesh_dim, dim=axis_idx))
                    return result

    return tensor


def _get_current_rank():
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def _get_current_device():
    if torch.cuda.is_available():
        local_rank = dist.get_rank() if dist.is_initialized() else 0
        return torch.device(f"cuda:{local_rank}")
    elif hasattr(torch, "mps") and torch.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _shard_tensor(tensor, layout, rank=None, device=None):
    if rank is None:
        rank = _get_current_rank()
    if device is None:
        device = _get_current_device()

    if hasattr(layout, 'backend_layout'):
        backend_layout = layout.backend_layout
    else:
        backend_layout = layout

    if backend_layout is None:
        return tensor

    axes = getattr(backend_layout, 'axes', None)
    if axes is None:
        axes = getattr(backend_layout, 'sharding', None)
    if axes is None:
        if isinstance(backend_layout, dict):
            axes = backend_layout.get('axes', [])
        else:
            axes = []

    device_mesh = getattr(backend_layout, 'device_mesh', None)
    if device_mesh is None:
        mesh = getattr(backend_layout, 'mesh', None)
        if mesh is not None:
            device_mesh = mesh

    sharding_axes = [ax for ax in axes if ax is not None]
    if not sharding_axes:
        return tensor

    first_sharding_axis = sharding_axes[0]

    mesh_dim_size = 1
    if device_mesh is not None:
        axis_names = getattr(device_mesh, 'axis_names', [])
        shape = getattr(device_mesh, 'shape', [])

        if first_sharding_axis in axis_names:
            axis_idx = axis_names.index(first_sharding_axis)
            mesh_dim_size = shape[axis_idx]

    dim_size = tensor.shape[first_sharding_axis]

    if mesh_dim_size > 1 and dim_size >= mesh_dim_size:
        if dim_size % mesh_dim_size == 0:
            shard_size = dim_size // mesh_dim_size
            shards = list(torch.split(tensor, shard_size, dim=first_sharding_axis))
        else:
            shards = list(torch.chunk(tensor, mesh_dim_size, dim=first_sharding_axis))

        shard = shards[rank % len(shards)]

        shard = shard.to(device)

        return shard
    else:
        return tensor


def distribute_variable(value, layout):
    from keras.src.distribution import TensorLayout

    if isinstance(layout, TensorLayout):
        backend_layout = layout.backend_layout if hasattr(layout, 'backend_layout') else layout
    else:
        backend_layout = layout

    should_shard = False
    if backend_layout is not None:
        axes = getattr(backend_layout, 'axes', [])
        if axes is not None:
            should_shard = any(ax is not None for ax in axes)

    if not should_shard:
        current_device = _get_current_device()
        if value.device != current_device:
            value = value.to(current_device)
        return value

    current_device = _get_current_device()
    if value.device.type in ('cuda', 'mps'):
        value = value.cpu()

    sharded_value = _shard_tensor(value, backend_layout, rank=_get_current_rank(), device=current_device)

    sharded_value._distributed_layout = backend_layout
    sharded_value._full_shape = value.shape
    sharded_value._sharding_axis = getattr(backend_layout, 'axes', [None] * len(value.shape))[0]
    sharded_value._is_sharded = True

    return sharded_value


def all_gather_variable(variable):
    if not getattr(variable, '_is_sharded', False):
        return variable

    if not dist.is_initialized():
        return variable

    full_shape = getattr(variable, '_full_shape', None)
    sharding_axis = getattr(variable, '_sharding_axis', 0)

    if full_shape is None:
        return variable

    world_size = dist.get_world_size()

    local_shard = variable
    if local_shard.device.type in ('cuda', 'mps'):
        local_shard = local_shard.cpu()

    output = torch.zeros(full_shape, dtype=local_shard.dtype, device='cpu')

    dim_size = full_shape[sharding_axis]
    shard_size = dim_size // world_size

    if not isinstance(local_shard, (list, tuple)):
        local_shard = [local_shard]

    output_list = [torch.zeros_like(s) for s in local_shard]
    dist.all_gather(output_list, local_shard[0])

    gathered = torch.cat(output_list, dim=sharding_axis)

    current_device = _get_current_device()
    gathered = gathered.to(current_device)

    return gathered


def distribute_data_input(per_process_batch, layout, batch_dim_name):

    if isinstance(per_process_batch, (tuple, list)):
        result = type(per_process_batch)(
            distribute_data_input(x, layout, batch_dim_name) for x in per_process_batch
        )
        return result

    result = per_process_batch
    return result


def initialize(job_addresses=None, num_processes=None, process_id=None):
    if job_addresses is None and "KERAS_DISTRIBUTION_JOB_ADDRESSES" in os.environ:
        job_addresses = os.environ["KERAS_DISTRIBUTION_JOB_ADDRESSES"]
    if num_processes is None and "KERAS_DISTRIBUTION_NUM_PROCESSES" in os.environ:
        num_processes = int(os.environ["KERAS_DISTRIBUTION_NUM_PROCESSES"])
    if process_id is None and "KERAS_DISTRIBUTION_PROCESS_ID" in os.environ:
        process_id = int(os.environ["KERAS_DISTRIBUTION_PROCESS_ID"])

    if num_processes is None or num_processes <= 1:
        return

    if job_addresses and "," in job_addresses:
        job_addresses = job_addresses.split(",")
        coordinator_address = job_addresses[0]
    else:
        coordinator_address = job_addresses

    os.environ["MASTER_ADDR"] = coordinator_address.split(":")[0] if ":" in coordinator_address else coordinator_address
    os.environ["MASTER_PORT"] = "29500"

    if "RANK" not in os.environ:
        os.environ["RANK"] = str(process_id if process_id is not None else 0)
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(process_id if process_id is not None else 0)
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = str(num_processes)

    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"

        dist.init_process_group(
            backend=backend,
            init_method="env://",
            rank=process_id if process_id is not None else 0,
            world_size=num_processes,
        )


def num_processes():
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def process_id():
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def _to_backend_mesh(device_mesh):
    return {
        "devices": device_mesh.devices,
        "axis_names": device_mesh.axis_names,
        "shape": device_mesh.shape,
    }


def _to_backend_layout(tensor_layout):
    if tensor_layout.device_mesh is None:
        raise ValueError(
            "Cannot create sharding when device mesh is not set "
            "for TensorLayout."
        )

    return {
        "axes": tensor_layout.axes,
        "mesh": _to_backend_mesh(tensor_layout.device_mesh),
    }


def all_reduce(tensor, reduce_op="sum", axis_name=None):
    """All-reduce a tensor across the mesh.

    Args:
        tensor: The tensor to reduce.
        reduce_op: The reduction operation ("sum", "product", "min", "max").
        axis_name: Optional axis name for JAX compatibility (unused in torch).

    Returns:
        The reduced tensor.
    """
    if not dist.is_initialized():
        return tensor

    if reduce_op == "sum":
        op = dist.ReduceOp.SUM
    elif reduce_op == "product":
        op = dist.ReduceOp.PRODUCT
    elif reduce_op == "min":
        op = dist.ReduceOp.MIN
    elif reduce_op == "max":
        op = dist.ReduceOp.MAX
    else:
        op = dist.ReduceOp.SUM

    dist.all_reduce(tensor, op)

    return tensor


def all_gather(tensor, axis=0, axis_name=None):
    """Gather tensors from all processes along the specified axis.

    Args:
        tensor: The tensor to gather.
        axis: The axis along which to concatenate gathered tensors.
        axis_name: Optional axis name for JAX compatibility (unused in torch).

    Returns:
        The gathered tensor with all shards concatenated along the axis.
    """
    if not dist.is_initialized():
        return tensor

    world_size = dist.get_world_size()

    if world_size == 1:
        return tensor

    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor)

    result = torch.cat(tensor_list, dim=axis)
    return result


def broadcast(tensor, src=0):
    if not dist.is_initialized():
        return tensor

    rank = dist.get_rank()

    if rank != src:
        tensor = torch.zeros_like(tensor)

    dist.broadcast(tensor, src=src)

    return tensor

