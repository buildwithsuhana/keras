import os

import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor import Replicate
from torch.distributed.tensor import Shard
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor._op_schema import OpSchema
from torch.distributed.tensor._op_schema import OpSpec
from torch.distributed.tensor._op_schema import OpStrategy
from torch.distributed.tensor._op_schema import RuntimeSchemaInfo

try:
    from torch.distributed.tensor._ops import register_op_strategy
except ImportError:
    try:
        from torch.distributed.tensor._ops.utils import register_op_strategy
    except ImportError:
        # Fallback for very old/new versions
        register_op_strategy = DTensor._op_dispatcher.sharding_propagator.register_op_strategy


def normalize_dim(dim, ndim):
    """Normalize a dimension index."""
    return dim if dim >= 0 else dim + ndim


def shift_shard_dims_after_remove(placements, remove_dim=0):
    """Shift sharded dimensions after removing a dimension."""
    new_placements = []
    for p in placements:
        if isinstance(p, Shard) and p.dim > remove_dim:
            new_placements.append(Shard(p.dim - 1))
        else:
            new_placements.append(p)
    return tuple(new_placements)


_UNBIND_REGISTERED = False


def _register_unbind_strategy():
    global _UNBIND_REGISTERED
    if _UNBIND_REGISTERED:
        return

    # Use the function call version of the decorator to register
    register_op_strategy(
        torch.ops.aten.unbind.int, schema_info=RuntimeSchemaInfo(1)
    )(_unbind_op_strategy)
    _UNBIND_REGISTERED = True


def list_devices(device_type=None):
    """List available devices."""
    device_type = device_type or "gpu"
    if torch.distributed.is_initialized():
        count = torch.distributed.get_world_size()
    elif "WORLD_SIZE" in os.environ:
        count = int(os.environ["WORLD_SIZE"])
    else:
        if device_type.lower() == "gpu":
            count = torch.cuda.device_count() or 1
        else:
            count = 1
    return [f"{device_type.lower()}:{i}" for i in range(count)]


def get_device_count(device_type=None):
    """Get the total number of available devices."""
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    elif "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    else:
        if device_type and device_type.lower() == "gpu":
            return torch.cuda.device_count() or 1
        else:
            return 1


def initialize(job_addresses=None, num_processes=None, process_id=None):
    """Initialize the distributed process group."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if not torch.distributed.is_initialized():
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "localhost"
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29500"
        backend = os.environ.get("TORCH_DISTRIBUTED_BACKEND") or (
            "nccl" if torch.cuda.is_available() else "gloo"
        )
        torch.distributed.init_process_group(backend=backend)


def num_processes():
    """Get the number of processes in the distributed group."""
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def process_id():
    """Get the rank of the current process."""
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def _to_backend_device(device_name):
    """Map a Keras device name to a Torch device."""
    if device_name is not None:
        device_name = device_name.lower()
        if "gpu" in device_name:
            device_name = device_name.replace("gpu", "cuda")
        return torch.device(device_name)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def _to_backend_mesh(keras_mesh):
    """Map a Keras `DeviceMesh` to a Torch `DeviceMesh`."""
    return init_device_mesh(
        "cuda" if torch.cuda.is_available() else "cpu",
        mesh_shape=tuple(keras_mesh.shape),
        mesh_dim_names=tuple(keras_mesh.axis_names),
    )


def _to_backend_layout(tensor_layout):
    """Map a Keras `TensorLayout` to Torch distribution specs."""
    if tensor_layout is None:
        return None

    keras_mesh = tensor_layout.device_mesh
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
    """Distribute a Torch tensor according to a layout."""
    if layout is None:
        return tensor

    from keras.src.distribution import distribution_lib as dist_lib

    dist = dist_lib.distribution()
    if not isinstance(dist, dist_lib.ModelParallel):
        return tensor

    _register_unbind_strategy()

    from keras.src.distribution import TensorLayout

    if isinstance(layout, TensorLayout):
        layout = _to_backend_layout(layout)
    if isinstance(layout, DTensorLayout):
        torch_mesh, placements = layout.device_mesh, layout.placements
    elif isinstance(layout, (list, tuple)):
        torch_mesh, placements = layout
    else:
        return tensor

    if isinstance(tensor, DTensor):
        return tensor.redistribute(
            device_mesh=torch_mesh, placements=placements
        )

    return torch.distributed.tensor.distribute_tensor(
        tensor, device_mesh=torch_mesh, placements=placements
    )


def distribute_variable(value, layout, trainable=True):
    """Create a distributed Torch parameter."""
    if layout is None:
        return torch.nn.Parameter(value, requires_grad=trainable)

    from keras.src.distribution import distribution_lib as dist_lib

    dist = dist_lib.distribution()
    if not isinstance(dist, dist_lib.ModelParallel):
        return torch.nn.Parameter(value, requires_grad=trainable)

    dtensor = distribute_tensor(value, layout)
    return torch.nn.Parameter(dtensor, requires_grad=trainable)


def distribute_data_input(tensor, layout, batch_dim_name):
    """Map local data to a distributed `DTensor`."""
    if layout is None:
        return tensor
    from keras.src.distribution import distribution_lib as dist_lib

    dist = dist_lib.distribution()
    if not isinstance(dist, dist_lib.ModelParallel):
        return tensor

    _register_unbind_strategy()

    if isinstance(tensor, DTensor):
        return tensor
    from keras.src.distribution import TensorLayout

    if isinstance(layout, TensorLayout):
        layout = _to_backend_layout(layout)

    if isinstance(layout, DTensorLayout):
        torch_mesh, placements = layout.device_mesh, layout.placements
    elif isinstance(layout, (list, tuple)):
        torch_mesh, placements = layout
    else:
        return tensor

    return DTensor.from_local(
        tensor, device_mesh=torch_mesh, placements=placements
    )


def _unbind_op_strategy(op_schema: OpSchema):
    """Registered sharding strategy for `torch.ops.aten.unbind.int`."""
    input_strategy = op_schema.args_schema[0]
    dim = op_schema.args_schema[1] if len(op_schema.args_schema) > 1 else 0
    dim = normalize_dim(dim, input_strategy.ndim)

    mesh = input_strategy.mesh
    new_strategy = OpStrategy([])

    for arg_strategy in input_strategy.strategies:
        arg_spec = arg_strategy.output_spec
        # Check if sharded along dim
        is_sharded = any(
            isinstance(p, Shard) and p.dim == dim for p in arg_spec.placements
        )

        if is_sharded:
            # If sharded along unbind dim, suggest replication.
            replicated_placements = tuple(
                Replicate() for _ in arg_spec.placements
            )
            replicated_spec = DTensorSpec(
                mesh=mesh,
                placements=replicated_placements,
                tensor_meta=arg_spec.tensor_meta,
            )
            output_placements = tuple(Replicate() for _ in arg_spec.placements)
            output_spec = DTensorSpec(
                mesh=mesh,
                placements=output_placements,
            )
            new_strategy.strategies.append(
                OpSpec(
                    output_specs=(output_spec,) * input_strategy.shape[dim],
                    input_specs=(replicated_spec,),
                )
            )
        else:
            # Not sharded along dim, forward sharding
            output_placements = shift_shard_dims_after_remove(
                arg_spec.placements, dim
            )
            output_spec = DTensorSpec(
                mesh=mesh,
                placements=tuple(output_placements),
            )
            new_strategy.strategies.append(
                OpSpec(
                    output_specs=(output_spec,) * input_strategy.shape[dim],
                    input_specs=(arg_spec,),
                )
            )
    return new_strategy
