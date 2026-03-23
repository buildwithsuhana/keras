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

from keras.src.utils.module_utils import torch_xla


def _get_torch_device_type(device_type):
    """Normalize device type to PyTorch conventions."""
    device_type = (device_type or "gpu").lower()
    if device_type == "gpu":
        return "cuda"
    if device_type == "tpu":
        return "xla"
    return device_type


def _get_device_count(device_type):
    """Return the number of available devices for a given type."""
    device_type = _get_torch_device_type(device_type)
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])

    if device_type == "cuda":
        return torch.cuda.device_count() or 1
    if device_type == "xla" and torch_xla.available:
        import torch_xla.core.xla_model as xm

        return len(xm.get_xla_supported_devices())
    return 1


def list_devices(device_type=None):
    """Return all the available devices based on the device type."""
    count = _get_device_count(device_type)
    device_type = _get_torch_device_type(device_type)
    return [f"{device_type}:{i}" for i in range(count)]


def get_device_count(device_type=None):
    """Returns the number of available devices."""
    return _get_device_count(device_type)


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

    if torch_xla.available:
        import torch_xla.distributed.xla_backend as xla_backend

        backend = "xla"
    else:
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
    from keras.src.backend.torch import core

    device_name = core.get_device()
    if "cuda" in device_name:
        device_type = "cuda"
    elif "mps" in device_name:
        device_type = "mps"
    elif "xpu" in device_name:
        device_type = "xpu"
    else:
        device_type = "cpu"

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
    torch_mesh = keras_mesh.backend_mesh or _to_backend_mesh(keras_mesh)

    placements = []
    for mesh_dim_name in keras_mesh.axis_names:
        shard_dim = None
        if tensor_layout.axes is not None:
            try:
                shard_dim = tensor_layout.axes.index(mesh_dim_name)
            except ValueError:
                shard_dim = None

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

    if isinstance(tensor, DTensor):
        return tensor.redistribute(
            device_mesh=layout.device_mesh, placements=layout.placements
        )

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


def _register_sharding_rules():
    """Register sharding rules for ops missing from PyTorch DTensor."""
    try:
        from torch.distributed.tensor._api import DTensor
        from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
        from torch.distributed.tensor._op_schema import (
            OpSchema,
            OutputSharding,
        )
        from torch.distributed.tensor.placement_types import Replicate
        
        # Robust import for registration function
        register_prop_rule = None
        try:
            from torch.distributed.tensor._ops.registration import register_prop_rule
        except ImportError:
            try:
                from torch.distributed.tensor.registration import register_prop_rule
            except ImportError:
                pass

        # Robust import for utils
        try:
            from torch.distributed.tensor._ops.utils import (
                is_tensor_dim_sharded,
                shift_shard_dims_after_remove,
            )
        except ImportError:
            # Fallback implementations if utils are missing
            def is_tensor_dim_sharded(spec, dim):
                from torch.distributed.tensor.placement_types import Shard
                return any(isinstance(p, Shard) and p.dim == dim for p in spec.placements)

            def shift_shard_dims_after_remove(placements, remove_dim):
                from torch.distributed.tensor.placement_types import Shard
                new_placements = []
                for p in placements:
                    if isinstance(p, Shard) and p.dim > remove_dim:
                        new_placements.append(Shard(p.dim - 1))
                    else:
                        new_placements.append(p)
                return new_placements

    except ImportError as e:
        return

    def unbind_rule(op_schema):
        try:
            input_spec = op_schema.args_schema[0]
            dim = op_schema.args_schema[1] if len(op_schema.args_schema) > 1 else 0
            if dim < 0:
                dim += input_spec.ndim


            if is_tensor_dim_sharded(input_spec, dim=dim):
                # Replicate the input if the unbind dimension is sharded
                replicate_spec = DTensorSpec(
                    mesh=input_spec.mesh,
                    placements=tuple([Replicate()] * len(input_spec.placements)),
                    tensor_meta=input_spec.tensor_meta,
                )
                return OutputSharding(
                    output_spec=None,
                    needs_redistribute=True,
                    redistribute_schema=OpSchema(
                        op=op_schema.op,
                        args_schema=(replicate_spec, dim),
                        kwargs_schema=op_schema.kwargs_schema,
                    ),
                )

            output_placements = tuple(
                shift_shard_dims_after_remove(input_spec.placements, dim)
            )
            output_shape = list(input_spec.shape)
            output_dim_size = output_shape.pop(dim)
            output_stride = list(input_spec.stride)
            output_stride.pop(dim)


            output_spec = [
                DTensorSpec(
                    mesh=input_spec.mesh,
                    placements=output_placements,
                    tensor_meta=TensorMeta(
                        shape=torch.Size(output_shape),
                        stride=tuple(output_stride),
                        dtype=input_spec.tensor_meta.dtype,
                    ),
                )
            ] * output_dim_size
            return OutputSharding(output_spec=tuple(output_spec))
        except Exception as e:
            return OutputSharding(None)

    def view_rule(op_schema):
        try:
            input_spec = op_schema.args_schema[0]
            if not isinstance(input_spec, DTensorSpec):
                return OutputSharding(None)

            # Check if any dimension is sharded
            # Check if any placement is not Replicate (Shard or Partial)
            is_sharded = any(not p.is_replicate() for p in input_spec.placements)
            
            if is_sharded:
                # Replicate before view to avoid strict_view errors in PyTorch.
                replicate_spec = DTensorSpec(
                    mesh=input_spec.mesh,
                    placements=tuple([Replicate()] * len(input_spec.placements)),
                    tensor_meta=input_spec.tensor_meta,
                )
                return OutputSharding(
                    output_spec=None,
                    needs_redistribute=True,
                    redistribute_schema=OpSchema(
                        op=op_schema.op,
                        args_schema=(replicate_spec,) + op_schema.args_schema[1:],
                        kwargs_schema=op_schema.kwargs_schema,
                    ),
                )
            
            # If already Replicated, return a fresh spec with correct placements.
            # We do NOT provide tensor_meta here so that the ShardingPropagator
            # can fill it in with the correct output shape/strides.
            return OutputSharding(
                output_spec=DTensorSpec(
                    mesh=input_spec.mesh,
                    placements=input_spec.placements
                )
            )
        except Exception:
            pass
        return OutputSharding(None)

    # Register for unbind
    try:
        sharding_propagator = DTensor._op_dispatcher.sharding_propagator
        overloads = torch.ops.aten.unbind.overloads()
        for overload_name in overloads:
            overload = getattr(torch.ops.aten.unbind, overload_name)
            try:
                if overload in sharding_propagator.op_strategy_funcs:
                    del sharding_propagator.op_strategy_funcs[overload]
                if register_prop_rule is not None:
                    register_prop_rule(overload)(unbind_rule)
                else:
                    sharding_propagator.register_sharding_prop_rule(
                        overload, unbind_rule
                    )
            except Exception:
                pass
    except Exception:
        pass

    # Register for view/reshape
    view_ops = [
        torch.ops.aten.view.default,
        torch.ops.aten.view.dtype,
        torch.ops.aten._unsafe_view.default,
        torch.ops.aten.reshape.default,
    ]
    for op in view_ops:
        try:
            if op in sharding_propagator.op_strategy_funcs:
                del sharding_propagator.op_strategy_funcs[op]
            if register_prop_rule is not None:
                register_prop_rule(op)(view_rule)
            else:
                sharding_propagator.register_sharding_prop_rule(
                    op, view_rule
                )
        except Exception:
            pass


_register_sharding_rules()




