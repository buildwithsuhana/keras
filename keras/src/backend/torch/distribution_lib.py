import os

import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor import Replicate
from torch.distributed.tensor import Shard


def get_device_count(device_type=None):
    """Returns total device count across all hosts."""
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    return torch.cuda.device_count() or 1


def list_devices(device_type=None):
    """Returns Keras device strings representing global indices."""
    device_type = device_type or "gpu"
    count = get_device_count(device_type)
    return [f"{device_type.lower()}:{i}" for i in range(count)]


def initialize(job_addresses=None, num_processes=None, process_id=None):
    """Initialize the current process for distributed training."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    torch.distributed.init_process_group(backend=backend)


def num_processes():
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def process_id():
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def _to_backend_device(device_name):
    """Returns the local device for the current process."""
    if device_name is not None:
        device_name_lower = device_name.lower()
        if "cpu" in device_name_lower:
            return torch.device("cpu")
        if "gpu" in device_name_lower or "cuda" in device_name_lower:
            if ":" in device_name_lower:
                device_idx = int(device_name_lower.split(":")[1])
                return torch.device(f"cuda:{device_idx}")
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            return torch.device(
                f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
            )

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def _to_backend_mesh(keras_mesh):
    """Maps a Keras DeviceMesh to a Torch DeviceMesh."""
    from torch.distributed.device_mesh import DeviceMesh as TorchDeviceMesh

    if isinstance(keras_mesh, TorchDeviceMesh):
        return keras_mesh
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    return init_device_mesh(
        device_type,
        mesh_shape=tuple(keras_mesh.shape),
        mesh_dim_names=tuple(keras_mesh.axis_names),
    )


class DTensorLayout:
    """Wraps a torch DeviceMesh + placements for use as a backend layout."""

    def __init__(self, device_mesh, placements):
        self.device_mesh = device_mesh
        self.placements = placements


def _to_backend_layout(tensor_layout):
    """Converts Keras TensorLayout to PyTorch (DeviceMesh, placements)."""
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


def distribute_tensor(tensor, layout):
    """Scatters or replicates a tensor according to the layout."""
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

    if isinstance(tensor, DTensor):
        return tensor.redistribute(
            device_mesh=layout.device_mesh, placements=layout.placements
        )

    return torch.distributed.tensor.distribute_tensor(
        tensor, device_mesh=layout.device_mesh, placements=layout.placements
    )


def distribute_variable(value, layout, trainable=True):
    """Wraps a distributed tensor as a Parameter."""
    dtensor = distribute_tensor(value, layout)
    return torch.nn.Parameter(dtensor, requires_grad=trainable)


def distribute_data_input(tensor, layout, batch_dim_name):
    """Wraps per-process data as a DTensor."""
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

    if isinstance(tensor, DTensor):
        return tensor

    if not isinstance(tensor, torch.Tensor):
        from keras.src.backend.common import global_state
        from keras.src.backend.torch import core as torch_core

        dist = global_state.get_global_attribute("distribution")
        global_state.set_global_attribute("distribution", None)
        try:
            tensor = torch_core.convert_to_tensor(tensor)
        finally:
            global_state.set_global_attribute("distribution", dist)

    if tensor.device.type == "meta":
        return tensor

    return DTensor.from_local(
        tensor, device_mesh=layout.device_mesh, placements=layout.placements
    )


_UNBIND_REGISTERED = False


def _unbind_op_strategy(op_schema):
    from torch.distributed.tensor._dtensor_spec import DTensorSpec
    from torch.distributed.tensor._op_schema import OpSpec
    from torch.distributed.tensor._op_schema import OpStrategy

    input_strategy = op_schema.args_schema[0]
    dim = op_schema.args_schema[1] if len(op_schema.args_schema) > 1 else 0
    dim = dim if dim >= 0 else dim + input_strategy.ndim

    mesh = input_strategy.mesh
    new_strategy = OpStrategy([])

    for arg_strategy in input_strategy.strategies:
        arg_spec = arg_strategy.output_spec
        is_sharded = any(
            isinstance(p, Shard) and p.dim == dim for p in arg_spec.placements
        )

        if is_sharded:
            rep_spec = DTensorSpec(
                mesh=mesh,
                placements=tuple(Replicate() for _ in arg_spec.placements),
                tensor_meta=arg_spec.tensor_meta,
            )
            out_spec = DTensorSpec(
                mesh=mesh,
                placements=tuple(Replicate() for _ in arg_spec.placements),
            )
            new_strategy.strategies.append(
                OpSpec(
                    output_specs=(out_spec,) * input_strategy.shape[dim],
                    input_specs=(rep_spec,),
                )
            )
        else:
            out_placements = []
            for p in arg_spec.placements:
                if isinstance(p, Shard) and p.dim > dim:
                    out_placements.append(Shard(p.dim - 1))
                else:
                    out_placements.append(p)
            out_spec = DTensorSpec(mesh=mesh, placements=tuple(out_placements))
            new_strategy.strategies.append(
                OpSpec(
                    output_specs=(out_spec,) * input_strategy.shape[dim],
                    input_specs=(arg_spec,),
                )
            )
    return new_strategy


def _register_unbind_strategy():
    """Registers sharding propagation for `unbind` (used by MHA heads)."""
    global _UNBIND_REGISTERED
    if _UNBIND_REGISTERED:
        return

    from torch.distributed.tensor._op_schema import RuntimeSchemaInfo
    from torch.distributed.tensor._ops import register_op_strategy

    register_op_strategy(
        torch.ops.aten.unbind.int, schema_info=RuntimeSchemaInfo(1)
    )(_unbind_op_strategy)
    _UNBIND_REGISTERED = True
