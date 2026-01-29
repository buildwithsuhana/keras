"""Utilities for distribution strategy with PyTorch backend.

This module implements distribution strategies for PyTorch using DTensor
(Distributed Tensor) API. It provides support for:
- Data Parallelism: Distribute data across devices while replicating model
- Model Parallelism: Shard model weights across devices

The implementation uses PyTorch's distributed module and DTensor for
efficient tensor sharding and communication.

Note: This implementation requires PyTorch 2.0+ with DTensor support.
For older versions, falls back to basic distributed data parallelism.
"""

import os
import re
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import DeviceMesh, Shard, replicate
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    ParallelStyle,
    RowwiseParallel,
)

from keras.src.backend.common import global_state
from keras.src.random import seed_generator
from keras.src.utils import rng_utils


def list_devices(device_type: Optional[str] = None) -> List[str]:
    """Return all the available devices based on the device type.

    Note that this should return the global devices in a distributed setting.

    Args:
        device_type: string of `"cpu"`, `"gpu"` or `"tpu"`. Defaults to `"gpu"`
            or `"tpu"` if available when device_type is not provided. Otherwise
            will return the `"cpu"` devices.

    Return:
        List of devices that are available for distribute computation.
    """
    device_type = device_type.lower() if device_type else None
    devices = []

    if device_type in (None, "gpu", "cuda"):
        # Check for CUDA GPUs
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            for i in range(num_gpus):
                devices.append(f"cuda:{i}")
        # Check for MPS (Apple Silicon)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            devices.append("mps:0")
        # Check for XPU (Intel)
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            num_xpus = torch.xpu.device_count()
            for i in range(num_xpus):
                devices.append(f"xpu:{i}")

    if device_type in (None, "tpu") and not devices:
        # Try to detect TPU
        try:
            import torch_xla.core.xla_model as xm
            if xm.xla_device():
                devices.append("tpu:0")
        except ImportError:
            pass

    if device_type == "cpu" or (device_type is None and not devices):
        # Return CPU devices
        devices = [f"cpu:{i}" for i in range(min(torch.get_num_threads(), 8))]

    return devices


def get_device_count(device_type: Optional[str] = None) -> int:
    """Returns the number of available PyTorch devices.

    Args:
        device_type: Optional device type to count (e.g., "cpu", "gpu", "tpu").
            If `None`, it defaults to counting "gpu" or "tpu" devices if
            available, otherwise it counts "cpu" devices. It does not
            return the sum of all device types.

    Returns:
        int: The total number of PyTorch devices for the specified type.
    """
    device_type = device_type.lower() if device_type else None

    if device_type in (None, "gpu", "cuda"):
        if torch.cuda.is_available():
            return torch.cuda.device_count()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return 1
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.xpu.device_count()

    if device_type in (None, "tpu"):
        try:
            import torch_xla.core.xla_model as xm
            if xm.xla_device():
                return xm.num_global_visible_devices()
        except ImportError:
            pass

    if device_type == "cpu":
        return min(torch.get_num_threads(), 8)

    # Fallback: return CPU count
    return min(torch.get_num_threads(), 8)


def _to_backend_device(device_name: str) -> torch.device:
    """Convert a device string to a torch.device.

    Args:
        device_name: Device string like "cpu:0", "cuda:0", "mps:0", "tpu:0"

    Returns:
        torch.device instance
    """
    if isinstance(device_name, torch.device):
        return device_name

    device_name = str(device_name)
    if ":" in device_name:
        device_type, device_id = device_name.split(":")
        device_type = device_type.lower()
        if device_type == "gpu":
            device_type = "cuda"
        return torch.device(f"{device_type}:{device_id}")
    else:
        return torch.device(device_name.lower())


def _parse_device_mesh_shape(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """Parse device mesh shape tuple."""
    return shape


def _get_device_mesh(devices: List[str]) -> DeviceMesh:
    """Create a DeviceMesh from a list of device strings.

    Args:
        devices: List of device strings like ["cuda:0", "cuda:1", ...]

    Returns:
        DeviceMesh instance
    """
    # Convert device strings to torch devices
    torch_devices = [_to_backend_device(d) for d in devices]

    # Determine mesh shape based on available devices
    if len(torch_devices) == 1:
        # Single device - use 1D mesh
        mesh_shape = (1,)
    elif len(torch_devices) <= 4:
        # Small number of devices - use 1D mesh
        mesh_shape = (len(torch_devices),)
    else:
        # Multiple devices - try to create 2D mesh
        import math

        num_devices = len(torch_devices)
        # Find factors for balanced mesh
        factors = []
        for i in range(2, int(math.sqrt(num_devices)) + 1):
            if num_devices % i == 0:
                factors.append((i, num_devices // i))
        if factors:
            # Use the most balanced factorization
            mesh_shape = factors[-1]
        else:
            mesh_shape = (num_devices,)

    # Reshape devices to mesh shape
    devices_array = torch_devices
    while len(devices_array) < mesh_shape[0]:
        devices_array = devices_array + devices_array[:1]

    try:
        mesh = DeviceMesh(
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            mesh=devices_array[: mesh_shape[0]],
            mesh_dim_names=[f"dim_{i}" for i in range(len(mesh_shape))],
        )
    except Exception:
        # Fallback to simple mesh
        mesh = DeviceMesh(
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            mesh=torch_devices,
            mesh_dim_names=["dim_0"],
        )

    return mesh


class TorchTensorLayout:
    """Internal wrapper for DTensor layout information.

    This class wraps PyTorch DTensor's mesh and placements to provide
    a Keras-compatible interface for tensor layouts.
    """

    def __init__(
        self,
        mesh: DeviceMesh,
        placements: Optional[List[Union[Shard, None]]] = None,
    ):
        self._mesh = mesh
        self._placements = placements or [None] * mesh.ndim

    @property
    def mesh(self) -> DeviceMesh:
        return self._mesh

    @property
    def placements(self) -> List[Union[Shard, None]]:
        return self._placements

    def __repr__(self):
        return f"TorchTensorLayout(mesh={self._mesh}, placements={self._placements})"


def distribute_tensor(tensor, layout):
    """Distribute the tensor based on the layout.

    Args:
        tensor: `torch.Tensor` that needs to be distributed.
        layout: TensorLayout for the distribution information, or a
            TorchTensorLayout instance.

    Returns:
        Distributed tensor (DTensor or sharded tensor).
    """
    from keras.src.distribution import TensorLayout

    if isinstance(layout, TensorLayout):
        layout = _to_backend_layout(layout)

    if isinstance(layout, TorchTensorLayout):
        try:
            # Try to use DTensor if available
            from torch.distributed.tensor import distribute_tensor as dtensor_dist

            if hasattr(torch.distributed.tensor, "DTensor"):
                # Create DTensor from tensor
                dtensor = dtensor_dist(
                    tensor,
                    device_mesh=layout.mesh,
                    placements=layout.placements,
                )
                return dtensor
        except (ImportError, Exception):
            pass

    # Fallback: manual sharding
    return _manual_shard_tensor(tensor, layout)


def distribute_variable(value, layout):
    """Create a distributed variable for PyTorch.

    For PyTorch, this creates a Parameter with the tensor distributed
    according to the layout specification.

    Args:
        value: the initial value of the variable (torch.Tensor or numpy array).
        layout: TensorLayout for the created variable, or a
            TorchTensorLayout instance.

    Returns:
        torch.nn.Parameter with distributed tensor.
    """
    from keras.src.distribution import TensorLayout

    if isinstance(layout, TensorLayout):
        layout = _to_backend_layout(layout)

    if isinstance(layout, TorchTensorLayout):
        try:
            # Try DTensor
            from torch.distributed.tensor import distribute_tensor as dtensor_dist

            tensor = distribute_tensor(
                torch.as_tensor(value),
                device_mesh=layout.mesh,
                placements=layout.placements,
            )
            return torch.nn.Parameter(tensor)
        except (ImportError, Exception):
            pass

    # Fallback: manual sharding
    tensor = distribute_tensor(torch.as_tensor(value), layout)
    return torch.nn.Parameter(tensor)


def _manual_shard_tensor(
    tensor: torch.Tensor, layout: TorchTensorLayout
) -> torch.Tensor:
    """Manually shard a tensor without DTensor.

    This is a fallback for environments where DTensor is not available.

    Args:
        tensor: The tensor to shard.
        layout: The sharding layout specification.

    Returns:
        Sharded tensor placed on appropriate devices.
    """
    mesh = layout.mesh
    placements = layout.placements

    # Get device from mesh
    device = mesh.device

    # If no sharding (all None placements), replicate
    if all(p is None for p in placements):
        return tensor.to(device).clone()

    # Find the first sharded dimension
    shard_dim = None
    for i, placement in enumerate(placements):
        if isinstance(placement, Shard):
            shard_dim = i
            break

    if shard_dim is None:
        return tensor.to(device).clone()

    # Calculate chunk size
    dim_size = tensor.shape[shard_dim]
    num_chunks = mesh.shape[0]  # First mesh dimension

    if num_chunks > dim_size:
        # More chunks than dimension size
        num_chunks = dim_size if dim_size > 0 else 1

    # Split tensor along shard dimension
    chunks = torch.chunk(tensor, num_chunks, dim=shard_dim)

    # Get the appropriate chunk for this device
    device_id = 0
    if hasattr(mesh, "_device_id"):
        device_id = mesh._device_id
    elif hasattr(mesh, "device_id"):
        device_id = mesh.device_id

    chunk_idx = device_id % len(chunks)
    sharded_tensor = chunks[chunk_idx].to(device)

    return sharded_tensor


def distribute_data_input(per_process_batch, layout, batch_dim_name):
    """Distribute the input data with the corresponding layout.

    Note that the inputs here is a local worker batch. Within the local worker,
    the data need to be further partitioned to map to each of the devices.

    Args:
        per_process_batch: `torch.Tensor` that is already sharded to
            a local process size.
        layout: `TensorLayout` for the distribution information, or a
            `TorchTensorLayout` instance.

    Returns:
        A global batch distributed according to `layout`.
    """
    from keras.src.distribution import TensorLayout

    if isinstance(layout, TensorLayout):
        layout = _to_backend_layout(layout)

    return distribute_tensor(per_process_batch, layout)


def initialize_rng():
    """Initializes the global random number generator across processes.

    This is required for consistent initialization in multi-host settings.
    """
    global_seed = rng_utils.get_random_seed()
    # Only set a random seed if not already set
    # via keras.config.set_random_seed()
    if global_seed is None:
        # Generate a random seed on each CPU host and psum them to get a single
        # consistent seed across all processes.
        if dist.is_available() and dist.is_initialized():
            # Use distributed reduction to get consistent seed
            local_seed = torch.tensor(
                [seed_generator.make_default_seed()],
                dtype=torch.int32,
            )
            global_seed_tensor = local_seed.clone()
            dist.all_reduce(global_seed_tensor, op=dist.ReduceOp.SUM)
            global_seed = int(global_seed_tensor.item()) % (2**32)
        else:
            global_seed = seed_generator.make_default_seed()

        # Set the global seed.
        rng_utils.set_random_seed(global_seed)

    # Check if the global seed generator is set and ensure it has an initialized
    # seed. Otherwise, reset the seed to the global seed.
    global_seed_generator = global_state.get_global_attribute(
        seed_generator.GLOBAL_SEED_GENERATOR
    )
    if global_seed_generator is not None:
        seed = global_seed_generator.get_config()["seed"]
        if seed is None:
            global_state.set_global_attribute(
                seed_generator.GLOBAL_SEED_GENERATOR,
                seed_generator.SeedGenerator(
                    seed=global_seed,
                    name=global_seed_generator.name,
                    backend=global_seed_generator.backend,
                ),
            )


def initialize(
    job_addresses: Optional[str] = None,
    num_processes: Optional[int] = None,
    process_id: Optional[int] = None,
):
    """Initialize the distributed environment.

    Args:
        job_addresses: string. Comma separated IP addresses for all the jobs
            that will form the whole computation cluster. For PyTorch backend,
            the backend will use torch.distributed.init_process_group with
            the NCCL backend for GPU or GLOO for CPU.
        num_processes: int. The number of worker/processes that will form the
            whole computation cluster.
        process_id: int. The ID number of the current worker/process. The value
            should be ranged from `0` to `num_processes - 1`.
    """
    if job_addresses and "," in job_addresses:
        job_addresses = job_addresses.split(",")
        coordinator_address = job_addresses[0]
    else:
        coordinator_address = job_addresses

    # Set environment variables
    if coordinator_address:
        os.environ["MASTER_ADDR"] = coordinator_address
    if num_processes is not None:
        os.environ["WORLD_SIZE"] = str(num_processes)
    if process_id is not None:
        os.environ["RANK"] = str(process_id)

    # Choose backend based on device type
    if torch.cuda.is_available():
        backend = "nccl"
    else:
        backend = "gloo"

    # Set default port if not set
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"

    # Initialize the process group
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available")
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)

    # Ensure the random number generator is initialized across processes.
    initialize_rng()


def num_processes() -> int:
    """Return the number of processes for the current distribution setting."""
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()


def process_id() -> int:
    """Return the current process ID for the distribution setting."""
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()


def _to_backend_mesh(device_mesh):
    """Convert the DeviceMesh to PyTorch backend specific Mesh.

    Args:
        device_mesh: DeviceMesh instance to convert.

    Returns:
        A `torch.distributed.DeviceMesh` instance.
    """
    shape = device_mesh.devices.shape
    devices = [_to_backend_device(d) for d in device_mesh.devices.flatten()]
    devices = torch.tensor(
        [d.index if hasattr(d, "index") else 0 for d in devices],
        dtype=torch.long,
    ).reshape(shape)

    # Create mesh with appropriate device
    mesh_device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Create the DeviceMesh
    torch_mesh = DeviceMesh(
        device=mesh_device,
        mesh=devices,
        mesh_dim_names=list(device_mesh.axis_names),
    )

    return torch_mesh


def _to_backend_layout(tensor_layout):
    """Convert the TensorLayout to PyTorch DTensor Layout.

    Args:
        tensor_layout: TensorLayout instance to convert.

    Returns:
        A `TorchTensorLayout` instance with DeviceMesh and placements.
    """
    from keras.src.distribution import TensorLayout

    if tensor_layout.device_mesh is None:
        raise ValueError(
            "Cannot create sharding when device mesh is not set "
            "for TensorLayout."
        )

    # Convert to PyTorch DeviceMesh
    torch_mesh = _to_backend_mesh(tensor_layout.device_mesh)

    # Convert axes to placements
    placements = []
    for axis in tensor_layout.axes:
        if axis is None:
            placements.append(None)
        else:
            # Find the mesh dimension index for this axis name
            axis_names = tensor_layout.device_mesh.axis_names
            if axis in axis_names:
                mesh_dim_idx = axis_names.index(axis)
                placements.append(Shard(dim=mesh_dim_idx))
            else:
                placements.append(None)

    return TorchTensorLayout(mesh=torch_mesh, placements=placements)


# Path separator adapter for converting Keras paths to PyTorch parameter paths
_KERAS_TO_PYTORCH_PATTERN = re.compile(r"(\w+)/(\w+)")


def convert_keras_path_to_pytorch(keras_path: str) -> str:
    """Convert Keras-style path (dense/kernel) to PyTorch-style (dense.weight).

    This adapter allows Keras regex patterns using `/` separators to work
    with PyTorch parameters that use `.` separators.

    Args:
        keras_path: Path string in Keras format, e.g., "dense/kernel",
            "conv2d/bias"

    Returns:
        Path string in PyTorch format, e.g., "dense.weight", "conv2d.bias"
    """
    # Handle common weight name patterns
    if keras_path.endswith("/kernel"):
        return keras_path[:-7] + ".weight"
    elif keras_path.endswith("/bias"):
        return keras_path[:-5] + ".bias"
    elif keras_path.endswith("/gamma"):
        return keras_path[:-6] + ".weight"  # LayerNorm gamma -> weight
    elif keras_path.endswith("/beta"):
        return keras_path[:-5] + ".bias"  # LayerNorm beta -> bias
    elif keras_path.endswith("/moving_mean"):
        return keras_path[:-13] + ".running_mean"
    elif keras_path.endswith("/moving_variance"):
        return keras_path[:-16] + ".running_var"
    else:
        # General case: replace / with .
        return keras_path.replace("/", ".")


def convert_pytorch_path_to_keras(pytorch_path: str) -> str:
    """Convert PyTorch-style path (dense.weight) to Keras-style (dense/kernel).

    Args:
        pytorch_path: Path string in PyTorch format, e.g., "dense.weight",
            "conv2d.bias"

    Returns:
        Path string in Keras format, e.g., "dense/kernel", "conv2d/bias"
    """
    # Handle common weight name patterns
    if pytorch_path.endswith(".weight"):
        return pytorch_path[:-7] + "/kernel"
    elif pytorch_path.endswith(".bias"):
        return pytorch_path[:-5] + "/bias"
    elif pytorch_path.endswith(".running_mean"):
        return pytorch_path[:-13] + "/moving_mean"
    elif pytorch_path.endswith(".running_var"):
        return pytorch_path[:-11] + "/moving_variance"
    else:
        # General case: replace . with /
        return pytorch_path.replace(".", "/")


class PathSeparatorAdapter:
    """Adapter to handle path separator differences between Keras and PyTorch.

    Keras uses '/' separators (e.g., 'dense/kernel') while PyTorch uses
    '.' separators (e.g., 'dense.weight'). This adapter provides methods
    to convert between the two formats and match patterns against both.
    """

    @staticmethod
    def keras_to_pytorch(keras_path: str) -> str:
        """Convert Keras path format to PyTorch format."""
        return convert_keras_path_to_pytorch(keras_path)

    @staticmethod
    def pytorch_to_keras(pytorch_path: str) -> str:
        """Convert PyTorch path format to Keras format."""
        return convert_pytorch_path_to_keras(pytorch_path)

    @staticmethod
    def match_pattern(pattern: str, path: str, try_both: bool = True) -> bool:
        """Try to match a pattern against a path, trying both formats.

        Args:
            pattern: The regex pattern (typically in Keras format with '/').
            path: The path to match against (could be in either format).
            try_both: Whether to also try matching with path converted.

        Returns:
            True if pattern matches, False otherwise.
        """
        # Try direct match first
        if re.search(pattern, path):
            return True

        if try_both:
            # Convert path to the other format and try again
            if "/" in path:
                # Path is in Keras format, convert to PyTorch
                pytorch_path = convert_keras_path_to_pytorch(path)
                if re.search(pattern, pytorch_path):
                    return True
            elif "." in path:
                # Path is in PyTorch format, convert to Keras
                keras_path = convert_pytorch_path_to_keras(path)
                if re.search(pattern, keras_path):
                    return True

        return False


# Utility function to wrap a model for distributed training
def prepare_model_for_distribution(
    model: nn.Module,
    device_mesh: DeviceMesh,
    parallel_style: str = "tensor_parallel",
) -> nn.Module:
    """Prepare a PyTorch model for distributed training.

    Args:
        model: The PyTorch model to prepare.
        device_mesh: The DeviceMesh for distribution.
        parallel_style: One of "tensor_parallel" or "sequence_parallel".

    Returns:
        Model prepared for distributed training.
    """
    from torch.distributed.tensor.parallel import (
        parallelize_module,
        SequenceParallel,
    )

    # Create parallel style
    if parallel_style == "tensor_parallel":
        plan = {
            "*": ColwiseParallel(),
        }
    elif parallel_style == "sequence_parallel":
        plan = SequenceParallel()
    else:
        plan = None

    if plan is not None:
        # Parallelize the model
        parallelize_module(model, device_mesh, plan)

    return model


# Utility function to get distributed data loader
def getDistributedDataLoader(
    dataset,
    batch_size: int,
    num_replicas: Optional[int] = None,
    rank: Optional[int] = None,
    shuffle: bool = True,
    seed: int = 0,
    drop_last: bool = False,
):
    """Get a distributed data loader.

    Args:
        dataset: The dataset to load.
        batch_size: Batch size per replica.
        num_replicas: Number of replicas (defaults to world size).
        rank: Current rank (defaults to local rank).
        shuffle: Whether to shuffle data.
        seed: Random seed for shuffling.
        drop_last: Whether to drop last incomplete batch.

    Returns:
        DistributedSampler and DataLoader.
    """
    if num_replicas is None:
        num_replicas = num_processes()
    if rank is None:
        rank = process_id()

    sampler = torch.distributed.DistributedSampler(
        dataset,
        num_replicas=num_replicas,
        rank=rank,
        shuffle=shuffle,
        seed=seed,
        drop_last=drop_last,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
        drop_last=drop_last,
    )

    return sampler, dataloader

