"""Utilities for distribution strategy with Torch backend."""

import importlib
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor import Partial
from torch.distributed.tensor import Replicate
from torch.distributed.tensor import Shard
from torch.distributed.tensor import distribute_tensor as distribute_tensor_torch

from keras.src.backend.torch.core import get_device


def list_devices(device_type=None):
    """Return all the available devices based on the device type.

    Note that this should return the global devices in a distributed setting.

    Args:
        device_type: string of `"cpu"`, `"gpu"` or `"tpu"`. Defaults to `"gpu"`
            or `"tpu"` if available when device_type is not provided. Otherwise
            will return the `"cpu"` devices.

    Returns:
        List of devices that are available for distribute computation.
    """
    if device_type is None:
        if torch.cuda.is_available():
            device_type = "cuda"
        elif "xla" in get_device():
            device_type = "xla"
        elif torch.backends.mps.is_available():
            device_type = "mps"
        else:
            device_type = "cpu"

    device_type = device_type.lower().replace("gpu", "cuda").replace("tpu", "xla")

    if device_type == "cuda":
        count = torch.cuda.device_count()
    elif device_type == "xla" and importlib.util.find_spec("torch_xla"):
        xm = importlib.import_module("torch_xla.core.xla_model")
        count = len(xm.get_xla_supported_devices())
    else:
        count = 1 if device_type in ("mps", "cpu") else 0

    if dist.is_initialized():
        count = dist.get_world_size()

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


def distribute_value(value, layout):
    """Distribute the value based on the layout."""
    return distribute_tensor(value, layout)


def distribute_variable(value, layout):
    """Create a distributed variable for Torch."""
    return distribute_tensor(value, layout)


def distribute_tensor(tensor, layout):
    """Distribute the tensor based on the layout.

    Args:
        tensor: The tensor to distribute.
        layout: `TensorLayout` instance.

    Returns:
        A distributed tensor (DTensor).
    """
    from keras.src.distribution import TensorLayout

    if not isinstance(layout, TensorLayout):
        return tensor

    torch_mesh = layout.device_mesh.backend_mesh
    placements = _get_placements(layout)

    if isinstance(tensor, DTensor):
        if (
            tensor.device_mesh == torch_mesh
            and tensor.placements == tuple(placements)
        ):
            return tensor
        return tensor.redistribute(torch_mesh, placements)

    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor, device=get_device())

    if get_device() == "meta":
        if not isinstance(tensor, DTensor):
            return distribute_tensor_torch(tensor, torch_mesh, placements)
        return tensor.redistribute(torch_mesh, placements)

    # Optimization: use from_local for pre-sharded or matching device tensors
    if not isinstance(tensor, DTensor) and tensor.device.type == torch_mesh.device_type:
        local_tensor = tensor
        can_use_local = True
        for i, placement in enumerate(placements):
            if isinstance(placement, Shard):
                shard_dim = placement.dim
                num_chunks = torch_mesh.shape[i]
                if local_tensor.shape[shard_dim] % num_chunks != 0:
                    can_use_local = False
                    break
                idx = torch_mesh.get_local_rank(i)
                local_tensor = torch.chunk(local_tensor, num_chunks, dim=shard_dim)[idx]
        if can_use_local:
            return DTensor.from_local(local_tensor, torch_mesh, placements)

    if tensor.device.type != torch_mesh.device_type:
        if tensor.is_meta:
            tensor = torch.empty_like(tensor, device=torch_mesh.device_type)
        else:
            tensor = tensor.to(torch_mesh.device_type)

    return distribute_tensor_torch(tensor, torch_mesh, placements)


def distribute_data_input(per_process_batch, layout, batch_dim_name):
    """Distribute the input data with the corresponding layout.

    Args:
        per_process_batch: The local batch of data.
        layout: `TensorLayout` instance.
        batch_dim_name: The name of the batch dimension.

    Returns:
        A distributed tensor (DTensor).
    """
    from keras.src.distribution import TensorLayout

    if not isinstance(layout, TensorLayout):
        return per_process_batch

    torch_mesh = layout.device_mesh.backend_mesh
    placements = _get_placements(layout)

    if not isinstance(per_process_batch, torch.Tensor):
        per_process_batch = torch.as_tensor(
            per_process_batch, device=get_device()
        )

    if per_process_batch.device.type != torch_mesh.device_type:
        per_process_batch = per_process_batch.to(torch_mesh.device_type)

    if isinstance(per_process_batch, DTensor):
        return per_process_batch.redistribute(torch_mesh, placements)

    return distribute_tensor_torch(per_process_batch, torch_mesh, placements)


def initialize_rng():
    """Initializes the global random number generator across processes."""
    from keras.src.utils import rng_utils

    if rng_utils.get_random_seed() is None and dist.is_initialized():
        rank = dist.get_rank()
        backend = dist.get_backend()
        device_type = "cuda" if backend == "nccl" else ("xla" if backend == "xla" else "cpu")
        
        if device_type == "cuda":
            torch.cuda.set_device(rank % torch.cuda.device_count())
        elif device_type == "xla" and importlib.util.find_spec("torch_xla"):
            xm = importlib.import_module("torch_xla.core.xla_model")
            device_type = xm.xla_device()
            
        seed_value = np.random.randint(0, 2**31) if rank == 0 else 0
        seed_tensor = torch.tensor([seed_value], dtype=torch.int64, device=device_type)
        dist.broadcast(seed_tensor, 0)
        rng_utils.set_random_seed(int(seed_tensor.item()))


def initialize(job_addresses, num_processes, process_id):
    """Initialize the distribution strategy.

    Args:
        job_addresses: A comma-separated list of coordinator addresses.
        num_processes: Total number of processes.
        process_id: Rank of the current process.
    """
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

    if num_processes:
        os.environ["WORLD_SIZE"] = str(num_processes)
    if process_id:
        os.environ["RANK"] = str(process_id)

    if torch.cuda.is_available():
        backend = "nccl"
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
    elif importlib.util.find_spec("torch_xla"):
        backend = "xla"
        xm = importlib.import_module("torch_xla.core.xla_model")
        xm.xla_device()
    else:
        backend = "gloo"

    dist.init_process_group(backend)
    initialize_rng()


def num_processes():
    """Return the number of processes for the current distribution setting."""
    return dist.get_world_size() if dist.is_initialized() else 1


def process_id():
    """Return the current process ID for the distribution setting."""
    return dist.get_rank() if dist.is_initialized() else 0


_GLOBAL_DISPATCH_MODE = None


def _to_backend_mesh(device_mesh):
    """Convert the DeviceMesh to Torch backend specific Mesh.

    Args:
        device_mesh: `DeviceMesh` instance.

    Returns:
        Torch `DeviceMesh` instance.
    """
    first_dev = device_mesh.devices.flatten()[0]
    device_type = first_dev.split(":")[0] if ":" in first_dev else first_dev
    
    if device_type == "mps":
        device_type = "cpu"
        
    torch_mesh = init_device_mesh(
        device_type,
        device_mesh.devices.shape,
        mesh_dim_names=device_mesh.axis_names
    )
    
    global _GLOBAL_DISPATCH_MODE
    if _GLOBAL_DISPATCH_MODE is None:
        _GLOBAL_DISPATCH_MODE = KerasTorchDispatchMode()
        _GLOBAL_DISPATCH_MODE.__enter__()
        
    return torch_mesh


def _to_backend_layout(layout):
    """Convert the TensorLayout to Torch backend specific Placements."""
    return _get_placements(layout)


def _maybe_distribute_input(x, distribution):
    """Distribute the input data if it's not already a DTensor."""
    from keras.src import tree
    
    def _distribute_if_tensor(t):
        if isinstance(t, torch.Tensor) and not isinstance(t, DTensor):
            layout = _get_data_layout(t.shape, distribution)
            return distribute_tensor(t, layout)
        return t
        
    return tree.map_structure(_distribute_if_tensor, x)


def _get_data_layout(shape, distribution):
    """Default data layout if not provided."""
    if hasattr(distribution, "get_data_layout"):
        return distribution.get_data_layout(shape)
        
    from keras.src.distribution import TensorLayout
    spec = [None] * len(shape)
    if distribution.batch_dim_name:
        spec[0] = distribution.batch_dim_name
    return TensorLayout(spec, distribution.device_mesh)


def _get_placements(layout):
    """Compute Torch placements for a given layout."""
    mesh = layout.device_mesh
    axes = layout.axes
    placements = []
    for mesh_axis_name in mesh.axis_names:
        idx = -1
        for i, axis_name in enumerate(axes):
            if axis_name == mesh_axis_name:
                idx = i
                break
        if idx != -1:
            placements.append(Shard(idx))
        else:
            placements.append(Replicate())
    return placements


def distribute_dataset(dataset, distribution):
    """Create a distributed dataset for Torch."""
    from keras.src.utils.module_utils import tensorflow as tf
    if tf.available and isinstance(dataset, tf.data.Dataset):
        return distribution.distribute_dataset(dataset)
    return dataset


def auto_parallelize(layer, distribution):
    """Automatically parallelize a layer using native Torch TP if possible.

    Args:
        layer: The layer to parallelize.
        distribution: `ModelParallel` instance.

    Returns:
        The parallelized layer.
    """
    from keras.src.layers import Dense, Embedding
    from keras.src.distribution import ModelParallel
    
    if not isinstance(distribution, ModelParallel):
        return layer
        
    parallelize_plan = {}
    mesh_axis = distribution.device_mesh.axis_names[0]
    
    if isinstance(layer, Dense):
        layout = distribution.get_variable_layout(layer.kernel)
        if layout and mesh_axis in layout.axes:
            idx = layout.axes.index(mesh_axis)
            parallelize_plan["weight"] = ColwiseParallel() if idx == 1 else RowwiseParallel()
            
    elif isinstance(layer, Embedding):
        layout = distribution.get_variable_layout(layer.embeddings)
        if layout and mesh_axis in layout.axes:
            idx = layout.axes.index(mesh_axis)
            parallelize_plan["weight"] = RowwiseParallel() if idx == 0 else ColwiseParallel()
            
    if parallelize_plan:
        from keras.src.utils.torch_utils import TorchModuleWrapper
        if isinstance(layer, TorchModuleWrapper):
            parallelize_module(
                layer.module, 
                distribution.device_mesh.backend_mesh, 
                parallelize_plan
            )
            
    return layer


def parallelize_module(module, device_mesh, parallelize_plan):
    """Parallelize a torch module using native Torch TP."""
    from torch.distributed.tensor.parallel import parallelize_module as tp
    return tp(module, device_mesh, parallelize_plan)


def RowwiseParallel(*args, **kwargs):
    """Native Torch RowwiseParallel placement."""
    from torch.distributed.tensor.parallel import RowwiseParallel as RP
    return RP(*args, **kwargs)


def ColwiseParallel(*args, **kwargs):
    """Native Torch ColwiseParallel placement."""
    from torch.distributed.tensor.parallel import ColwiseParallel as CP
    return CP(*args, **kwargs)


def SequenceParallel(*args, **kwargs):
    """Native Torch SequenceParallel placement."""
    from torch.distributed.tensor.parallel import SequenceParallel as SP
    return SP(*args, **kwargs)


def _maybe_replicate(x):
    """Redistribute a DTensor to Replicate if it is sharded or partial."""
    if isinstance(x, DTensor):
        is_replicated = True
        for p in x.placements:
            if not isinstance(p, Replicate):
                is_replicated = False
                break
        if not is_replicated:
            return x.redistribute(x.device_mesh, [Replicate()] * x.device_mesh.ndim)
    return x


from torch.utils._python_dispatch import TorchDispatchMode
from keras.src import tree

class KerasTorchDispatchMode(TorchDispatchMode):
    """Dispatch mode to handle DTensor sharding errors by automatically replicating.

    This mode catches errors from operations that do not yet support distributed
    tensors (e.g., sharded or partial tensors) and automatically redistributes
    the inputs to a replicated state before retrying.
    """

    def _handle_sdpa(func, args, kwargs):
        # query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None
        query = args[0]
        key = args[1]
        value = args[2]
        attn_mask = kwargs.get("attn_mask", args[3] if len(args) > 3 else None)
        dropout_p = kwargs.get("dropout_p", args[4] if len(args) > 4 else 0.0)
        is_causal = kwargs.get("is_causal", args[5] if len(args) > 5 else False)
        scale = kwargs.get("scale", args[6] if len(args) > 6 else None)

        if attn_mask is not None:
            if hasattr(query, "device_mesh") and not hasattr(
                attn_mask, "device_mesh"
            ):
                attn_mask = distribute_tensor_torch(
                    attn_mask,
                    query.device_mesh,
                    [Replicate()] * query.device_mesh.ndim,
                )

        backends = [torch.nn.attention.SDPBackend.MATH]
        if not hasattr(query, "device_mesh"):
            backends.append(torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION)

        with torch.nn.attention.sdpa_kernel(backends=backends):
            return torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
            )

    def _handle_cross_entropy(func, args, kwargs):
        # input, target, weight=None, size_average=None, ignore_index=-100,
        # reduction='mean', label_smoothing=0.0
        input = args[0]
        target = args[1]
        reduction = kwargs.get("reduction", "mean")

        # Use safer manual cross entropy for sharded DTensors to avoid view errors
        if reduction == "none":
            log_prob = torch.nn.functional.log_softmax(input, dim=-1)
            if input.ndim == target.ndim + 1:
                # sparse cross entropy
                from keras.src.backend.torch.nn import one_hot

                target = one_hot(target, input.shape[-1], axis=-1)
            return -torch.sum(target * log_prob, dim=-1)

        return func(*args, **kwargs)

    _OP_HANDLERS = {}

    # Initialize _OP_HANDLERS with safe lookups
    def _init_handlers(self):
        handlers = {
            # SDPA ops
            "scaled_dot_product_attention": KerasTorchDispatchMode._handle_sdpa,
            "_scaled_dot_product_attention": KerasTorchDispatchMode._handle_sdpa,
            "_scaled_dot_product_attention_math": KerasTorchDispatchMode._handle_sdpa,
            # NLL Loss ops
            "nll_loss_forward": KerasTorchDispatchMode._handle_cross_entropy,
            "nll_loss": KerasTorchDispatchMode._handle_cross_entropy,
        }
        
        # Add aten ops
        for op_name, handler in handlers.items():
            if hasattr(torch.ops.aten, op_name):
                op = getattr(torch.ops.aten, op_name)
                if hasattr(op, "default"):
                    KerasTorchDispatchMode._OP_HANDLERS[op.default] = handler
                else:
                    KerasTorchDispatchMode._OP_HANDLERS[op] = handler
                    
        # Add functional ops
        KerasTorchDispatchMode._OP_HANDLERS[torch.nn.functional.scaled_dot_product_attention] = KerasTorchDispatchMode._handle_sdpa
        KerasTorchDispatchMode._OP_HANDLERS[torch.nn.functional.cross_entropy] = KerasTorchDispatchMode._handle_cross_entropy
        KerasTorchDispatchMode._OP_HANDLERS[torch.nn.functional.nll_loss] = KerasTorchDispatchMode._handle_cross_entropy

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if not self._OP_HANDLERS:
            self._init_handlers()
            
        if func in self._OP_HANDLERS:
            return self._OP_HANDLERS[func](func, args, kwargs or {})

        try:
            return func(*args, **(kwargs or {}))
        except (RuntimeError, NotImplementedError) as e:
            msg = str(e).lower()
            if any(
                k in msg
                for k in (
                    "sharding",
                    "shard",
                    "redistribute",
                    "partial",
                    "dtensor",
                    "flatten",
                    "view",
                    "reshape",
                    "boolean value",
                    "meta tensor",
                )
            ):
                # Fallback Stage 1: replicate all sharded inputs
                def _replicate_if_sharded(x):
                    if isinstance(x, DTensor):
                        is_replicated = True
                        for p in x.placements:
                            if not isinstance(p, Replicate):
                                is_replicated = False
                                break
                        if not is_replicated:
                            return x.redistribute(
                                x.device_mesh,
                                [Replicate()] * x.device_mesh.ndim,
                            )
                    return x

                new_args = tree.map_structure(_replicate_if_sharded, args)
                new_kwargs = tree.map_structure(
                    _replicate_if_sharded, kwargs or {}
                )

                try:
                    return func(*new_args, **new_kwargs)
                except (RuntimeError, NotImplementedError):
                    # Fallback Stage 2: convert to local tensors and ensure contiguity
                    def _to_local_contiguous(x):
                        if isinstance(x, DTensor):
                            return x.to_local().contiguous()
                        if isinstance(x, torch.Tensor):
                            return x.contiguous()
                        return x

                    local_args = tree.map_structure(_to_local_contiguous, new_args)
                    local_kwargs = tree.map_structure(_to_local_contiguous, new_kwargs)
                    res = func(*local_args, **local_kwargs)

                    # If we had DTensors, try to wrap the result back if possible
                    has_dt = any(
                        isinstance(x, DTensor) for x in tree.flatten(new_args)
                    )
                    if (
                        has_dt
                        and isinstance(res, torch.Tensor)
                        and not isinstance(res, DTensor)
                    ):
                        # Use the first DTensor's mesh and Replicate placements
                        dt = next(
                            x
                            for x in tree.flatten(new_args)
                            if isinstance(x, DTensor)
                        )
                        return DTensor.from_local(
                            res,
                            dt.device_mesh,
                            [Replicate()] * dt.device_mesh.ndim,
                        )
                    return res
            raise e
