import os
import re

import torch
import torch.distributed as dist

from keras.src.backend.torch.distribution_logger import get_logger


def list_devices(device_type=None):
    """Return all available devices based on the device type.

    In a distributed setting, returns the global list of devices.

    Args:
        device_type: Optional string specifying the device type. One of
            `"cpu"`, `"gpu"` or `"cuda"`. If not provided, returns all
            available GPU/MPS devices, or CPU devices if GPU is unavailable.

    Returns:
        List of device strings available for distributed computation.
            For CUDA, returns `"cuda:{i}"` for each GPU.
            For MPS, returns `["mps:0"]`.
            For CPU, returns `["cpu:0"]`.
    """
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

    if device_type is None:
        return ["cpu:0"]

    raise ValueError(f"Unsupported device_type: {device_type}")


def get_device_count(device_type=None):
    """Return the number of available devices of the specified type.

    Args:
        device_type: Optional string specifying the device type. One of
            `"cpu"`, `"gpu"` or `"cuda"`. If not provided, counts
            GPU/MPS devices if available, otherwise CPU devices.

    Returns:
        int: The number of available devices of the specified type.
            Returns 0 for GPU/MPS if no such devices are available.
    """
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


def _get_current_rank():
    """Get the current process rank in the distributed setting.

    Returns:
        int: The rank of the current process. Returns 0 if distributed
            is not initialized.
    """
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def _get_current_device():
    """Get the current device for this process in the distributed setting.

    Returns:
        torch.device: The current device. Returns cuda device with local
            rank for GPU, mps for Apple Silicon, or cpu otherwise.
    """
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return torch.device(f"cuda:{local_rank}")
    elif hasattr(torch, "mps") and torch.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _to_dtensor_device(device_name):
    """Convert device name string to torch.device for DTensor.

    Args:
        device_name: String device name (e.g., "cuda:0", "cpu")

    Returns:
        torch.device: The device for DTensor placement.
    """
    device_name = str(device_name)
    if "cuda" in device_name.lower():
        if ":" in device_name:
            device_type, device_id = device_name.split(":")
            return torch.device(f"cuda:{int(device_id)}")
        return torch.device("cuda")
    elif "mps" in device_name.lower():
        return torch.device("mps")
    return torch.device("cpu")


def _to_dtensor_mesh(device_mesh):
    """Convert a DeviceMesh to PyTorch DTensor DeviceMesh.

    Args:
        device_mesh: A DeviceMesh object with devices, axis_names, and shape.

    Returns:
        torch.distributed.tensor.DeviceMesh: The DTensor DeviceMesh.
    """
    from torch.distributed.tensor import DeviceMesh

    devices = device_mesh.devices.flatten().tolist()

    dtensor_mesh = DeviceMesh(
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        mesh=devices,
        mesh_dim_names=device_mesh.axis_names,
    )

    return dtensor_mesh


def _to_dtensor_placements(axes, device_mesh):
    """Convert Keras axis layout to DTensor placements.

    Args:
        axes: Tuple of axis names (strings or None) from TensorLayout.
        device_mesh: The DeviceMesh object.

    Returns:
        tuple: DTensor placements (Shard, Replicate, or Partial).
    """
    from torch.distributed.tensor import Replicate
    from torch.distributed.tensor import Shard

    placements = []
    for axis in axes:
        if axis is None:
            placements.append(Replicate())
        elif axis == "batch":
            placements.append(Shard(axis=0))
        else:
            if axis in device_mesh.axis_names:
                axis_idx = device_mesh.axis_names.index(axis)
                placements.append(Shard(axis=axis_idx))
            else:
                placements.append(Replicate())

    return tuple(placements)


def _is_dtensor_available():
    """Check if PyTorch DTensor is available.

    Returns:
        bool: True if DTensor is available, False otherwise.
    """
    return hasattr(torch.distributed, "tensor")


def _adapt_path(path):
    """Adapt a variable path between Keras and PyTorch formats.

    Keras uses '/' separators (e.g., 'dense/kernel') while PyTorch uses
    '.' (e.g., 'dense.weight'). This function normalizes paths for matching.

    Args:
        path: The variable path string.

    Returns:
        dict: A dict with 'keras' and 'torch' formats of the path.
    """
    return {
        "keras": path,
        "torch": path.replace("/", "."),
    }


def _match_layout_map_key(variable_path, layout_map):
    """Match a variable path against layout map keys, handling both separators.

    This adapter checks both Keras ('/') and PyTorch ('.') formats when
    matching layout_map keys to ensure compatibility with both naming
    conventions.

    Args:
        variable_path: The variable path (e.g., 'dense/kernel').
        layout_map: The LayoutMap with regex keys.

    Returns:
        The matching TensorLayout or None.
    """
    if variable_path in layout_map:
        return layout_map[variable_path]

    torch_path = variable_path.replace("/", ".")
    if torch_path in layout_map:
        return layout_map[torch_path]

    matching_keys = []
    for k in layout_map:
        if re.search(k, variable_path):
            matching_keys.append(k)

    for k in layout_map:
        torch_k = k.replace("/", ".")
        if re.search(torch_k, torch_path) and k not in matching_keys:
            matching_keys.append(k)

    if len(matching_keys) > 1:
        raise ValueError(
            f"Path '{variable_path}' matches multiple layout "
            f"specification keys: {matching_keys}. Please make "
            "sure each tensor/variable path only matches at most "
            "one layout specification key in the LayoutMap."
        )
    elif len(matching_keys) == 1:
        return layout_map[matching_keys[0]]
    return None


def distribute_tensor(tensor, layout):
    """Distribute a tensor according to the specified layout using DTensor.

    Args:
        tensor: The tensor to distribute.
        layout: The layout specification. Can be a TensorLayout object,
            a dict with 'axes' and 'device_mesh' keys, or None.

    Returns:
        For sharded tensors, returns a DTensor. For non-sharded tensors
        or when no sharding is specified, returns the original tensor.
    """
    logger = get_logger()
    rank = _get_current_rank()
    
    from keras.src.distribution import TensorLayout

    if isinstance(layout, TensorLayout):
        backend_layout = (
            layout.backend_layout
            if hasattr(layout, "backend_layout")
            else layout
        )
    else:
        backend_layout = layout

    if backend_layout is None:
        logger.debug(f"distribute_tensor: No layout specified, returning original tensor "
                    f"shape={tensor.shape}", rank)
        return tensor

    if hasattr(backend_layout, "axes"):
        axes = backend_layout.axes
    else:
        axes = backend_layout

    device_mesh = None
    if (
        hasattr(backend_layout, "device_mesh")
        and backend_layout.device_mesh is not None
    ):
        device_mesh = backend_layout.device_mesh

    sharding_axes = [ax for ax in axes if ax is not None]
    if not sharding_axes:
        logger.debug(f"distribute_tensor: No sharding axes specified, returning original tensor "
                    f"shape={tensor.shape}", rank)
        return tensor

    logger.info(f"distribute_tensor: Starting tensor distribution - "
                f"tensor_shape={tensor.shape}, sharding_axes={sharding_axes}, "
                f"device_mesh_shape={device_mesh.shape if device_mesh else None}", rank)

    if _is_dtensor_available():
        from torch.distributed.tensor import DTensor

        if isinstance(tensor, DTensor):
            logger.debug(f"distribute_tensor: Tensor is already a DTensor, returning as-is", rank)
            return tensor

        dtensor_mesh = _to_dtensor_mesh(device_mesh)
        placements = _to_dtensor_placements(axes, device_mesh)

        from torch.distributed.tensor import distribute_tensor as dtensor_dist

        result = dtensor_dist(tensor, dtensor_mesh, placements)
        logger.info(f"distribute_tensor: Distributed tensor to DTensor - "
                    f"original_shape={tensor.shape}, "
                    f"placements={[str(p) for p in placements]}", rank)
        return result
    else:
        first_sharding_axis = sharding_axes[0]

        if device_mesh is not None:
            axis_names = device_mesh.axis_names
            if first_sharding_axis in axis_names:
                axis_idx = axis_names.index(first_sharding_axis)
                mesh_dim = device_mesh.shape[axis_idx]

                if tensor.shape[axis_idx] >= mesh_dim:
                    if tensor.shape[axis_idx] % mesh_dim == 0:
                        shard_size = tensor.shape[axis_idx] // mesh_dim
                        result = list(
                            torch.split(tensor, shard_size, dim=axis_idx)
                        )
                        logger.info(f"distribute_tensor: Sharded tensor evenly - "
                                    f"original_shape={tensor.shape}, "
                                    f"shard_size={shard_size}, "
                                    f"num_shards={len(result)}, "
                                    f"axis={axis_idx}", rank)
                        return result
                    else:
                        result = list(
                            torch.chunk(tensor, mesh_dim, dim=axis_idx)
                        )
                        logger.info(f"distribute_tensor: Chunked tensor unevenly - "
                                    f"original_shape={tensor.shape}, "
                                    f"num_chunks={len(result)}, "
                                    f"axis={axis_idx}", rank)
                        return result

        logger.debug(f"distribute_tensor: No sharding applied, returning original tensor", rank)
        return tensor


def distribute_variable(value, layout):
    """Distribute a variable according to the specified layout using DTensor.

    Args:
        value: The variable (torch.Tensor) to distribute.
        layout: The layout specification. Can be a TensorLayout object,
            a dict with 'axes' and 'device_mesh' keys, or None.

    Returns:
        The distributed variable. Returns DTensor if sharding is needed,
        otherwise returns the variable on the current device.
    """
    logger = get_logger()
    rank = _get_current_rank()
    
    from keras.src.distribution import TensorLayout

    if isinstance(layout, TensorLayout):
        backend_layout = (
            layout.backend_layout
            if hasattr(layout, "backend_layout")
            else layout
        )
    else:
        backend_layout = layout

    should_shard = False
    if backend_layout is not None:
        axes = getattr(backend_layout, "axes", [])
        if axes is not None:
            should_shard = any(ax is not None for ax in axes)

    if not should_shard:
        current_device = _get_current_device()
        if value.device != current_device:
            value = value.to(current_device)
        logger.debug(f"distribute_variable: No sharding needed, moved to device - "
                    f"shape={value.shape}, device={current_device}", rank)
        return value

    logger.info(f"distribute_variable: Starting variable distribution - "
                f"shape={value.shape}, axes={axes}", rank)

    if _is_dtensor_available():
        from torch.distributed.tensor import DTensor

        if isinstance(value, DTensor):
            logger.debug(f"distribute_variable: Variable is already a DTensor", rank)
            return value

        device_mesh = getattr(backend_layout, "device_mesh", None)
        dtensor_mesh = _to_dtensor_mesh(device_mesh)
        axes = getattr(backend_layout, "axes", [])
        placements = _to_dtensor_placements(axes, device_mesh)

        from torch.distributed.tensor import distribute_tensor as dtensor_dist

        sharded_value = dtensor_dist(value, dtensor_mesh, placements)
        sharded_value._distributed_layout = backend_layout
        sharded_value._is_sharded = True

        logger.info(f"distribute_variable: Distributed variable to DTensor - "
                    f"original_shape={value.shape}, "
                    f"placements={[str(p) for p in placements]}", rank)
        return sharded_value
    else:
        current_device = _get_current_device()
        if value.device.type in ("cuda", "mps"):
            value = value.cpu()
            logger.debug(f"distribute_variable: Moved variable to CPU for sharding", rank)

        sharded_value = _shard_tensor(
            value,
            backend_layout,
            rank=_get_current_rank(),
            device=current_device,
        )

        sharded_value._distributed_layout = backend_layout
        sharded_value._full_shape = value.shape
        sharded_value._sharding_axis = getattr(
            backend_layout, "axes", [None] * len(value.shape)
        )[0]
        sharded_value._is_sharded = True

        logger.info(f"distribute_variable: Sharded variable - "
                    f"original_shape={value.shape}, "
                    f"sharded_shape={sharded_value.shape}, "
                    f"sharding_axis={sharded_value._sharding_axis}", rank)
        return sharded_value


def _shard_tensor(tensor, layout, rank=None, device=None):
    """Shard a tensor according to the specified layout.

    This is the original implementation, kept as fallback.

    Args:
        tensor: The tensor to shard.
        layout: The sharding layout specification.
        rank: Optional int, the current process rank. Defaults to
            _get_current_rank().
        device: Optional torch.device, the target device. Defaults to
            _get_current_device().

    Returns:
        The sharded tensor portion for the current rank and device.
    """
    logger = get_logger()
    
    if rank is None:
        rank = _get_current_rank()
    if device is None:
        device = _get_current_device()

    if hasattr(layout, "backend_layout"):
        backend_layout = layout.backend_layout
    else:
        backend_layout = layout

    if backend_layout is None:
        logger.debug(f"_shard_tensor: No layout, returning original tensor", rank)
        return tensor

    axes = getattr(backend_layout, "axes", None)
    if axes is None:
        axes = getattr(backend_layout, "sharding", None)
    if axes is None:
        if isinstance(backend_layout, dict):
            axes = backend_layout.get("axes", [])
        else:
            axes = []

    device_mesh = getattr(backend_layout, "device_mesh", None)
    if device_mesh is None:
        mesh = getattr(backend_layout, "mesh", None)
        if mesh is not None:
            device_mesh = mesh

    sharding_axes = [ax for ax in axes if ax is not None]
    if not sharding_axes:
        logger.debug(f"_shard_tensor: No sharding axes, returning original tensor", rank)
        return tensor

    first_sharding_axis = sharding_axes[0]

    mesh_dim_size = 1
    if device_mesh is not None:
        axis_names = getattr(device_mesh, "axis_names", [])
        shape = getattr(device_mesh, "shape", [])

        if first_sharding_axis in axis_names:
            axis_idx = axis_names.index(first_sharding_axis)
            mesh_dim_size = shape[axis_idx]

    dim_size = tensor.shape[first_sharding_axis]

    if mesh_dim_size > 1 and dim_size >= mesh_dim_size:
        if dim_size % mesh_dim_size == 0:
            shard_size = dim_size // mesh_dim_size
            shards = list(
                torch.split(tensor, shard_size, dim=first_sharding_axis)
            )
            logger.info(f"_shard_tensor: Even sharding - "
                        f"tensor_shape={tensor.shape}, shard_size={shard_size}, "
                        f"num_shards={len(shards)}, axis={first_sharding_axis}, "
                        f"mesh_dim={mesh_dim_size}", rank)
        else:
            shards = list(
                torch.chunk(tensor, mesh_dim_size, dim=first_sharding_axis)
            )
            logger.info(f"_shard_tensor: Uneven chunking - "
                        f"tensor_shape={tensor.shape}, num_chunks={len(shards)}, "
                        f"axis={first_sharding_axis}, mesh_dim={mesh_dim_size}", rank)

        shard = shards[rank % len(shards)]

        shard = shard.to(device)

        logger.debug(f"_shard_tensor: Selected shard for rank {rank} - "
                    f"shard_index={rank % len(shards)}, shard_shape={shard.shape}", rank)
        return shard
    else:
        logger.debug(f"_shard_tensor: No sharding needed (mesh_dim={mesh_dim_size}, "
                    f"dim_size={dim_size}), returning original tensor", rank)
        return tensor


def all_gather_variable(variable):
    """Gather all shards of a sharded variable back into a full tensor.

    Args:
        variable: The sharded variable to gather. Should have _is_sharded
            attribute set to True, or be a DTensor.

    Returns:
        The full gathered tensor if variable is sharded and distributed
            is initialized. Otherwise, returns the variable unchanged.
    """
    logger = get_logger()
    rank = _get_current_rank()
    
    if hasattr(variable, "_is_sharded") and variable._is_sharded:
        if _is_dtensor_available():
            from torch.distributed.tensor import DTensor

            if isinstance(variable, DTensor):
                logger.debug(f"all_gather_variable: Gathering DTensor full tensor", rank)
                return variable.full_tensor()

    if not getattr(variable, "_is_sharded", False):
        logger.debug(f"all_gather_variable: Variable not sharded, returning as-is", rank)
        return variable

    if not dist.is_initialized():
        logger.debug(f"all_gather_variable: Distributed not initialized, returning as-is", rank)
        return variable

    full_shape = getattr(variable, "_full_shape", None)
    sharding_axis = getattr(variable, "_sharding_axis", 0)

    if full_shape is None:
        logger.debug(f"all_gather_variable: No full_shape found, returning as-is", rank)
        return variable

    logger.info(f"all_gather_variable: Starting gather - "
                f"local_shape={variable.shape}, full_shape={full_shape}, "
                f"sharding_axis={sharding_axis}, world_size={dist.get_world_size()}", rank)

    local_shard = variable
    if local_shard.device.type in ("cuda", "mps"):
        local_shard = local_shard.cpu()
        logger.debug(f"all_gather_variable: Moved local shard to CPU", rank)
        
    if not isinstance(local_shard, (list, tuple)):
        local_shard = [local_shard]

    world_size = dist.get_world_size()
    output_list = [torch.empty_like(local_shard[0]) for _ in range(world_size)]
    dist.all_gather(output_list, local_shard[0])

    gathered = torch.cat(output_list, dim=sharding_axis)

    current_device = _get_current_device()
    gathered = gathered.to(current_device)

    logger.info(f"all_gather_variable: Gather complete - "
                f"gathered_shape={gathered.shape}, original_full_shape={full_shape}", rank)
    return gathered


def distribute_data_input(per_process_batch, layout, batch_dim_name):
    """Distribute input data according to the specified layout.

    Args:
        per_process_batch: The input batch (or tuple/list of batches) for
            the current process.
        layout: The data layout specification.
        batch_dim_name: The name of the batch dimension in the device mesh.

    Returns:
        The distributed input data. For tuples/lists, applies distribution
            recursively to each element. Returns the input unchanged otherwise.
    """
    logger = get_logger()
    rank = _get_current_rank()
    
    from keras.src.distribution import TensorLayout

    if isinstance(per_process_batch, (tuple, list)):
        logger.debug(f"distribute_data_input: Recursively distributing batch tuple/list", rank)
        result = type(per_process_batch)(
            distribute_data_input(x, layout, batch_dim_name)
            for x in per_process_batch
        )
        return result

    if layout is None:
        logger.debug(f"distribute_data_input: No layout specified, returning batch as-is", rank)
        return per_process_batch

    if isinstance(layout, TensorLayout):
        backend_layout = (
            layout.backend_layout
            if hasattr(layout, "backend_layout")
            else layout
        )
    else:
        backend_layout = layout

    if backend_layout is None:
        logger.debug(f"distribute_data_input: No backend_layout, returning batch as-is", rank)
        return per_process_batch

    if not _is_dtensor_available():
        logger.debug(f"distribute_data_input: DTensor not available, returning batch as-is", rank)
        return per_process_batch

    from torch.distributed.tensor import DTensor

    if isinstance(per_process_batch, DTensor):
        logger.debug(f"distribute_data_input: Batch is already a DTensor, returning as-is", rank)
        return per_process_batch

    axes = getattr(backend_layout, "axes", [])
    device_mesh = getattr(backend_layout, "device_mesh", None)

    if device_mesh is None or axes is None:
        logger.debug(f"distribute_data_input: No device_mesh or axes, returning batch as-is", rank)
        return per_process_batch

    logger.info(f"distribute_data_input: Distributing input data - "
                f"batch_shape={per_process_batch.shape}, axes={axes}, "
                f"batch_dim_name={batch_dim_name}", rank)

    dtensor_mesh = _to_dtensor_mesh(device_mesh)
    placements = _to_dtensor_placements(axes, device_mesh)

    from torch.distributed.tensor import distribute_tensor as dtensor_dist

    result = dtensor_dist(per_process_batch, dtensor_mesh, placements)
    logger.debug(f"distribute_data_input: Distributed batch to DTensor", rank)
    return result


def initialize(job_addresses=None, num_processes=None, process_id=None):
    """Initialize the distributed backend for multi-process training.

    Sets up the PyTorch distributed environment for multi-host/GPU training.
    This function should be called before any distributed computations.

    Parameters can be provided as arguments or via environment variables:
    - KERAS_DISTRIBUTION_JOB_ADDRESSES: Comma-separated IP addresses
    - KERAS_DISTRIBUTION_NUM_PROCESSES: Number of processes
    - KERAS_DISTRIBUTION_PROCESS_ID: Current process ID

    Args:
        job_addresses: Comma-separated IP addresses of all processes in
            the cluster. Can be just the coordinator address for some
            backends. Defaults to KERAS_DISTRIBUTION_JOB_ADDRESSES env var.
        num_processes: Total number of processes in the cluster. Defaults
            to KERAS_DISTRIBUTION_NUM_PROCESSES env var.
        process_id: ID of the current process (0 to num_processes-1).
            Defaults to KERAS_DISTRIBUTION_PROCESS_ID env var.

    Raises:
        RuntimeError: If CUDA is requested but not available.
    """
    logger = get_logger()
    
    if (
        job_addresses is None
        and "KERAS_DISTRIBUTION_JOB_ADDRESSES" in os.environ
    ):
        job_addresses = os.environ["KERAS_DISTRIBUTION_JOB_ADDRESSES"]
    if (
        num_processes is None
        and "KERAS_DISTRIBUTION_NUM_PROCESSES" in os.environ
    ):
        num_processes = int(os.environ["KERAS_DISTRIBUTION_NUM_PROCESSES"])
    if process_id is None and "KERAS_DISTRIBUTION_PROCESS_ID" in os.environ:
        process_id = int(os.environ["KERAS_DISTRIBUTION_PROCESS_ID"])

    if num_processes is None or num_processes <= 1:
        logger.debug("initialize: Single process mode, skipping distributed initialization")
        return

    logger.info(f"initialize: Starting distributed initialization - "
                f"job_addresses={job_addresses}, num_processes={num_processes}, "
                f"process_id={process_id}")

    if job_addresses and "," in job_addresses:
        job_addresses = job_addresses.split(",")
        coordinator_address = job_addresses[0]
    else:
        coordinator_address = job_addresses

    os.environ["MASTER_ADDR"] = (
        coordinator_address.split(":")[0]
        if ":" in coordinator_address
        else coordinator_address
    )
    os.environ["MASTER_PORT"] = "29500"

    if "RANK" not in os.environ:
        os.environ["RANK"] = str(process_id if process_id is not None else 0)
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = str(num_processes)

    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        
        logger.info(f"initialize: Initializing process group with backend={backend}, "
                    f"rank={process_id if process_id is not None else 0}, "
                    f"world_size={num_processes}")

        dist.init_process_group(
            backend=backend,
            init_method="env://",
            rank=process_id if process_id is not None else 0,
            world_size=num_processes,
        )
        
        logger.info(f"initialize: Distributed initialization complete - "
                    f"rank={dist.get_rank()}, world_size={dist.get_world_size()}")


def num_processes():
    """Get the number of processes in the distributed setting.

    Returns:
        int: The world size (number of processes). Returns 1 if
            distributed is not initialized.
    """
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def process_id():
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
            "max". Defaults to "sum".
        axis_name: The name of the mesh axis to reduce over.
            Defaults to "model".

    Returns:
        The reduced tensor.
    """
    logger = get_logger()
    rank = _get_current_rank()
    
    if not dist.is_initialized():
        logger.debug(f"all_reduce: Distributed not initialized, returning tensor as-is", rank)
        return tensor

    logger.info(f"all_reduce: Starting reduce operation - "
                f"tensor_shape={tensor.shape}, op={op}, axis_name={axis_name}, "
                f"world_size={dist.get_world_size()}", rank)

    if op == "sum":
        reduce_op = dist.ReduceOp.SUM
    elif op == "product":
        reduce_op = dist.ReduceOp.PRODUCT
    elif op == "min":
        reduce_op = dist.ReduceOp.MIN
    elif op == "max":
        reduce_op = dist.ReduceOp.MAX
    else:
        reduce_op = dist.ReduceOp.SUM

    dist.all_reduce(tensor, reduce_op)

    logger.debug(f"all_reduce: Reduce complete", rank)
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
    """
    logger = get_logger()
    rank = _get_current_rank()
    
    if not dist.is_initialized():
        logger.debug(f"all_gather: Distributed not initialized, returning tensor as-is", rank)
        return tensor

    world_size = dist.get_world_size()

    if world_size == 1:
        logger.debug(f"all_gather: Single process, returning tensor as-is", rank)
        return tensor

    logger.info(f"all_gather: Starting gather operation - "
                f"tensor_shape={tensor.shape}, axis={axis}, axis_name={axis_name}, "
                f"world_size={world_size}", rank)

    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor)

    result = torch.cat(tensor_list, dim=axis)

    logger.info(f"all_gather: Gather complete - "
                f"local_shape={tensor.shape}, gathered_shape={result.shape}, "
                f"concat_axis={axis}", rank)
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
    logger = get_logger()
    rank = _get_current_rank()
    
    if not dist.is_initialized():
        logger.debug(f"broadcast: Distributed not initialized, returning tensor as-is", rank)
        return tensor

    logger.info(f"broadcast: Starting broadcast - "
                f"tensor_shape={tensor.shape}, src={src}, "
                f"current_rank={dist.get_rank()}", rank)

    current_rank = dist.get_rank()

    if current_rank != src:
        tensor = torch.zeros_like(tensor)
        logger.debug(f"broadcast: Non-source rank {current_rank}, zeroing tensor", rank)

    dist.broadcast(tensor, src=src)

    logger.debug(f"broadcast: Broadcast complete", rank)
    return tensor
