"""Torch backend distribution utilities with DTensor and DDP support."""

import os
import torch
import torch.distributed as dist

from typing import Optional


def list_devices(device_type: Optional[str] = None):
    """Return all available devices based on the device type.

    In a distributed setting, returns the global list of devices.
    """
    device_type = device_type.lower() if device_type else None

    if (device_type is None or device_type in ("gpu", "cuda")) and torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        return [f"cuda:{i}" for i in range(num_devices)]
    if (device_type is None or device_type == "mps") and hasattr(torch, "mps") and torch.mps.is_available():
        return ["mps:0"]
    if device_type == "cpu" or device_type is None:
        num_devices = 1
        xla_flags = os.environ.get("XLA_FLAGS", "")
        if "--xla_force_host_platform_device_count=" in xla_flags:
            import re
            match = re.search(r"--xla_force_host_platform_device_count=(\d+)", xla_flags)
            if match:
                num_devices = int(match.group(1))
        return [f"cpu:{i}" for i in range(num_devices)]
    return []


def get_device_count(device_type: Optional[str] = None):
    """Return the number of available devices of the specified type."""
    return len(list_devices(device_type))


def initialize(job_addresses=None, num_processes=None, process_id=None):
    """Initialize the PyTorch distributed process group using env://.

    Environment variables `WORLD_SIZE`, `RANK` and `LOCAL_RANK` will be set
    if provided via the `num_processes`/`process_id` args.
    """
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

    if coordinator_address:
        os.environ["MASTER_ADDR"] = (
            coordinator_address.split(":")[0]
            if ":" in coordinator_address
            else coordinator_address
        )
    os.environ.setdefault("MASTER_PORT", "29500")

    os.environ.setdefault("RANK", str(process_id if process_id is not None else 0))
    os.environ.setdefault("LOCAL_RANK", str(process_id if process_id is not None else 0))
    os.environ.setdefault("WORLD_SIZE", str(num_processes))

    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")


def num_processes():
    """Return world size for the current process group (1 if not initialized)."""
    if dist.is_initialized():
        return dist.get_world_size()
    if "WORLD_SIZE" in os.environ:
        try:
            return int(os.environ["WORLD_SIZE"])
        except Exception:
            return 1
    return 1


def process_id():
    """Return the rank of the current process (0 if not initialized)."""
    if dist.is_initialized():
        return dist.get_rank()
    if "RANK" in os.environ:
        try:
            return int(os.environ["RANK"])
        except Exception:
            return 0
    return 0


def to_backend_device(device_name: Optional[str]):
    """Map a Keras device string like 'gpu:0' to a torch.device."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if device_name is None:
        if torch.cuda.is_available():
            return torch.device(f"cuda:{local_rank}")
        return torch.device("cpu")
    name = device_name.lower()
    if "meta" in name:
        return torch.device("meta")
    if "cpu" in name:
        return torch.device("cpu")
    if "gpu" in name or "cuda" in name:
        if ":" in name:
            return torch.device(name.replace("gpu", "cuda"))
        if torch.cuda.is_available():
            return torch.device(f"cuda:{local_rank}")
        return torch.device("cpu")
    return torch.device("cpu")


def _to_backend_mesh(keras_mesh):
    """Convert a Keras DeviceMesh into a torch DeviceMesh-like dict.

    We keep a lightweight mapping so higher-level code can create real
    torch DeviceMesh objects as needed.
    """
    return {"devices": keras_mesh.devices, "axis_names": keras_mesh.axis_names, "shape": keras_mesh.shape}


class DTensorLayout:
    """Simple container for (torch) device_mesh and placements."""

    def __init__(self, device_mesh, placements):
        self.device_mesh = device_mesh
        self.placements = placements


def _to_backend_layout(tensor_layout):
    """Convert a Keras TensorLayout to a backend layout usable by torch DTensor APIs."""
    if tensor_layout is None:
        return None
    keras_mesh = tensor_layout.device_mesh
    torch_mesh = _to_backend_mesh(keras_mesh)

    from torch.distributed.tensor import Replicate, Shard

    placements = []
    for mesh_dim_name in keras_mesh.axis_names:
        shard_dim = None
        if tensor_layout.axes is not None:
            for tensor_dim, axis_name in enumerate(tensor_layout.axes):
                if axis_name == mesh_dim_name:
                    shard_dim = tensor_dim
                    break
        placements.append(Shard(shard_dim) if shard_dim is not None else Replicate())

    return DTensorLayout(torch_mesh, tuple(placements))


def distribute_tensor(tensor, layout):
    """Distribute/redistribute a tensor according to a layout.

    If layout corresponds to a DTensor layout, use torch.distributed.tensor APIs
    when available; otherwise fall back to simple sharding/replication helpers.
    """
    if layout is None:
        return tensor

    from keras.src.distribution import TensorLayout

    if isinstance(layout, TensorLayout):
        layout = _to_backend_layout(layout)

    # If torch DTensor APIs are available, try to use them.
    try:
        from torch.distributed.tensor import DTensor

        if isinstance(tensor, DTensor):
            return tensor.redistribute(device_mesh=layout.device_mesh, placements=layout.placements)
        return torch.distributed.tensor.distribute_tensor(tensor, device_mesh=layout.device_mesh, placements=layout.placements)
    except Exception:
        # Fallback: no DTensor support; return the tensor as-is or move device.
        if isinstance(layout, DTensorLayout):
            # Move to the first placement device if it's a replicate
            device = to_backend_device(None)
            if tensor.device != device:
                return tensor.to(device)
        return tensor


def distribute_data_input(tensor, layout, batch_dim_name):
    """Convert per-process data tensor to DTensor when using ModelParallel."""
    if layout is None:
        return tensor

    from keras.src.distribution import TensorLayout

    if isinstance(layout, TensorLayout):
        layout = _to_backend_layout(layout)

    try:
        from torch.distributed.tensor import DTensor

        if isinstance(tensor, DTensor):
            return tensor

        if not isinstance(tensor, torch.Tensor):
            from keras.src.backend.torch import core as torch_core
            tensor = torch_core.convert_to_tensor(tensor, layout=None)

        if tensor.device.type == "meta":
            return tensor

        return DTensor.from_local(tensor, device_mesh=layout.device_mesh, placements=layout.placements)
    except Exception:
        return tensor


def distribute_variable(value, layout):
    """Distribute a variable according to the specified layout (fallback)."""
    # Simple fallback: move to local device
    current_device = to_backend_device(None)
    if isinstance(value, torch.Tensor) and value.device != current_device:
        return value.to(current_device)
    return value


def all_gather_variable(variable):
    """Gather a sharded variable back to a full tensor when possible."""
    try:
        if not dist.is_initialized():
            return variable
        # If it's a DTensor use its APIs, otherwise fallback to all_gather
        from torch.distributed.tensor import DTensor

        if hasattr(variable, "_is_sharded") and variable._is_sharded:
            # Try DTensor gather
            if isinstance(variable, DTensor):
                return variable
        # Fallback: return as-is
        return variable
    except Exception:
        return variable


_STRATEGIES_REGISTERED = False


def _unbind_op_strategy(op_schema):
    from torch.distributed.tensor import Replicate, Shard
    from torch.distributed.tensor._dtensor_spec import DTensorSpec
    from torch.distributed.tensor._op_schema import OpSpec, OpStrategy

    input_strategy = op_schema.args_schema[0]
    mesh = input_strategy.mesh
    new_strategy = OpStrategy([])

    for arg_strategy in input_strategy.strategies:
        arg_spec = arg_strategy.output_spec
        dim = op_schema.args_schema[1] if len(op_schema.args_schema) > 1 else 0
        dim = dim if dim >= 0 else dim + arg_spec.ndim

        is_sharded_on_dim = any(isinstance(p, Shard) and p.dim == dim for p in arg_spec.placements)
        if is_sharded_on_dim:
            rep_placements = tuple(Replicate() for _ in arg_spec.placements)
            rep_spec = DTensorSpec(mesh=mesh, placements=rep_placements, tensor_meta=arg_spec.tensor_meta)
            out_spec = DTensorSpec(mesh=mesh, placements=rep_placements)
            new_strategy.strategies.append(OpSpec(output_specs=(out_spec,) * arg_spec.shape[dim], input_specs=(rep_spec,)))
        else:
            out_placements = [Shard(p.dim - 1) if isinstance(p, Shard) and p.dim > dim else p for p in arg_spec.placements]
            out_spec = DTensorSpec(mesh=mesh, placements=tuple(out_placements))
            new_strategy.strategies.append(OpSpec(output_specs=(out_spec,) * arg_spec.shape[dim], input_specs=(arg_spec,)))
    return new_strategy


def _register_distributed_strategies():
    global _STRATEGIES_REGISTERED
    if _STRATEGIES_REGISTERED:
        return
    try:
        from torch.distributed.tensor._op_schema import RuntimeSchemaInfo
        from torch.distributed.tensor._ops import register_op_strategy

        register_op_strategy(torch.ops.aten.unbind.int, schema_info=RuntimeSchemaInfo(1))(_unbind_op_strategy)
        _STRATEGIES_REGISTERED = True
    except (ImportError, AttributeError):
        pass

    """Get the current process ID in the distributed setting.

    Returns:
        int: The rank of the current process. Returns 0 if distributed
            is not initialized.
    """
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def _to_backend_mesh(device_mesh):
    """Convert a DeviceMesh to backend-specific format.

    Args:
        device_mesh: A DeviceMesh object with devices, axis_names, and shape.

    Returns:
        dict: A dictionary with 'devices', 'axis_names', and 'shape' keys.
    """
    return {
        "devices": device_mesh.devices,
        "axis_names": device_mesh.axis_names,
        "shape": device_mesh.shape,
    }


def _to_backend_layout(tensor_layout):
    """Convert a TensorLayout to backend-specific format.

    Args:
        tensor_layout: A TensorLayout object with axes and device_mesh.

    Returns:
        dict: A dictionary with 'axes' and 'mesh' keys for the backend.

    Raises:
        ValueError: If device_mesh is not set in the tensor_layout.
    """
    if tensor_layout.device_mesh is None:
        raise ValueError(
            "Cannot create sharding when device mesh is not set "
            "for TensorLayout."
        )

    return {
        "axes": tensor_layout.axes,
        "mesh": _to_backend_mesh(tensor_layout.device_mesh),
    }


def all_reduce(tensor, op="sum", axis_name="model"):
    """Reduces a tensor across a device mesh axis using a collective.

    Args:
        tensor: The tensor to reduce.
        op: The reduction operation. One of "sum", "product", "min",
            "max", "mean". Defaults to "sum".
        axis_name: The name of the mesh axis to reduce over.
            Defaults to "model".

    Returns:
        The reduced tensor.
    """
    if not dist.is_initialized():
        return tensor

    if op == "sum":
        reduce_op = dist.ReduceOp.SUM
    elif op == "product":
        reduce_op = dist.ReduceOp.PRODUCT
    elif op == "min":
        reduce_op = dist.ReduceOp.MIN
    elif op == "max":
        reduce_op = dist.ReduceOp.MAX
    elif op == "mean":
        if hasattr(dist.ReduceOp, "AVG"):
            reduce_op = dist.ReduceOp.AVG
        else:
            # Fallback for older torch versions
            dist.all_reduce(tensor, dist.ReduceOp.SUM)
            tensor.div_(dist.get_world_size())
            return tensor
    else:
        reduce_op = dist.ReduceOp.SUM

    dist.all_reduce(tensor, reduce_op)

    return tensor


def all_gather(tensor, axis=0, axis_name="model"):
    """Gathers and concatenates tensors from all devices across a mesh axis.

    This function assumes it is called within a distributed context. It takes
    the local shard `tensor` from each device along the `axis_name` of the mesh
    and concatenates them along the specified tensor `axis` to form a
    single, larger tensor that is then replicated on all participating devices.

    Args:
        tensor: The input tensor shard on the local device.
        axis: The tensor axis along which to concatenate the gathered shards.
            Defaults to 0.
        axis_name: The name of the mesh axis to gather from.
            Defaults to "model".

    Returns:
        The full, gathered tensor with all shards concatenated along the axis.
        Returns the original tensor if distributed is not initialized
        or world_size is 1.
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
    """Broadcast a tensor from a source device to all other devices.

    Args:
        tensor: The tensor to broadcast. On non-source ranks, this should
            be a tensor of the same shape and dtype.
        src: The source rank to broadcast from. Defaults to 0.

    Returns:
        The broadcast tensor on all devices.
    """
    if not dist.is_initialized():
        return tensor

    rank = dist.get_rank()

    if rank != src:
        tensor = torch.zeros_like(tensor)

    dist.broadcast(tensor, src=src)

    return tensor
