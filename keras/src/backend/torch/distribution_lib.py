"""Distribution utilities for PyTorch backend using DTensor.

This module provides distribution strategy support for the PyTorch backend
using PyTorch's DTensor API. It enables both data parallelism and model
parallelism similar to how JAX handles distributed operations.

Key features:
- DTensor-based variable and tensor distribution
- Automatic path format conversion (Keras / vs PyTorch .)
- Model parallelism with tensor parallel styles
- Data parallelism with distributed sampling
"""

import os
import re
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor as torch_distribute_tensor
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

# Try to import DTensor placements, handle older PyTorch versions
try:
    from torch.distributed.tensor import Placement, Replicate, Shard
except ImportError:
    # For older PyTorch versions, define fallback classes
    class Placement:
        pass

    class Replicate:
        pass

    class Shard:
        def __init__(self, dim=0):
            self.dim = dim


# Global state for distribution
_distributed_initialized = False
_device_mesh_cache = {}


def _is_multi_process_mode():
    """Check if we're in multi-process distributed mode."""
    return (
        dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size() > 1
    )


def list_devices(device_type: Optional[str] = None) -> List[str]:
    """Return all available devices based on device type.

    Note: In a distributed setting, this returns global devices.

    Args:
        device_type: String of "cpu", "gpu" or "tpu". Defaults to "gpu" or
            "tpu" if available when device_type is not provided. Otherwise
            will return the "cpu" devices.

    Returns:
        List of devices available for distributed computation.
    """
    device_type = device_type.lower() if device_type else None

    devices = []
    if device_type in (None, "gpu"):
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            devices.extend([f"cuda:{i}" for i in range(num_gpus)])
    elif device_type == "cpu":
        # For CPU, we need distributed context
        if _is_multi_process_mode():
            num_procs = dist.get_world_size()
            for i in range(num_procs):
                devices.append(f"cpu:{i}")
        else:
            # Return a single CPU device if not distributed
            devices.append("cpu:0")

    if device_type == "tpu":
        # PyTorch doesn't have native TPU support like JAX
        pass

    # If no devices found and device_type was specified, return empty list
    if not devices and device_type is not None:
        return devices

    # If no devices found and device_type was None, return what we have
    if not devices:
        devices = ["cpu:0"]

    return devices


def get_device_count(device_type: Optional[str] = None) -> int:
    """Returns the number of available devices.

    Args:
        device_type: Optional device type to count (e.g., "cpu", "gpu", "tpu").
            If `None`, it defaults to counting "gpu" devices if available,
            otherwise it counts "cpu" devices. It does not return the sum
            of all device types.

    Returns:
        int: The total number of devices for the specified type.
    """
    device_type = device_type.lower() if device_type else None

    if device_type in (None, "gpu"):
        if torch.cuda.is_available():
            return torch.cuda.device_count()
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.xpu.device_count()

    if device_type == "cpu":
        if _is_multi_process_mode():
            return dist.get_world_size()
        return 1

    if device_type == "tpu":
        return 0

    return 0


def distribute_variable(value: torch.Tensor, layout) -> torch.Tensor:
    """Create a distributed variable using DTensor.

    This function distributes a tensor according to the specified layout
    using PyTorch's DTensor API. The tensor is scattered across devices
    according to the placement specifications.

    Args:
        value: The initial tensor value for the variable.
        layout: TensorLayout specifying the distribution.

    Returns:
        A DTensor distributed according to the layout.
    """
    # Avoid circular imports
    from keras.src.distribution import TensorLayout

    if layout is None:
        return value

    # In multi-process mode, skip DTensor distribution
    # Each process creates variables locally and uses DDP for gradient sync
    if _is_multi_process_mode():
        return value

    if isinstance(layout, TensorLayout):
        device_mesh = layout.device_mesh.backend_mesh
        placements = _tensor_layout_to_placements(layout)
    else:
        # Assume it's already a backend-compatible layout
        device_mesh = layout.device_mesh.backend_mesh
        placements = _tensor_layout_to_placements(layout)

    # Use DTensor to distribute the variable
    return torch_distribute_tensor(value, device_mesh, placements)


def distribute_tensor(tensor: torch.Tensor, layout) -> torch.Tensor:
    """Distribute a tensor based on the layout.

    This function can be used both in eager context and within
    torch.no_grad() contexts.

    Args:
        tensor: The tensor to distribute.
        layout: TensorLayout specifying the distribution.

    Returns:
        A DTensor distributed according to the layout.
    """
    # Avoid circular imports
    from keras.src.distribution import TensorLayout

    if layout is None:
        return tensor

    # In multi-process mode, skip DTensor distribution
    # Each process handles tensors locally
    if _is_multi_process_mode():
        return tensor

    if isinstance(layout, TensorLayout):
        device_mesh = layout.device_mesh.backend_mesh
        placements = _tensor_layout_to_placements(layout)
    else:
        device_mesh = layout.device_mesh.backend_mesh
        placements = _tensor_layout_to_placements(layout)

    return torch_distribute_tensor(tensor, device_mesh, placements)


def _tensor_layout_to_placements(layout) -> List[Placement]:
    """Convert TensorLayout axes to DTensor placements.

    Args:
        layout: TensorLayout instance.

    Returns:
        List of Placement objects for DTensor.
    """
    placements = []
    for axis in layout.axes:
        if axis is None:
            placements.append(Replicate())
        elif isinstance(axis, str):
            # Map axis names to placements
            axis_lower = axis.lower()
            if axis_lower == "batch":
                placements.append(Shard(0))
            elif axis_lower == "model":
                # For model parallelism, use Shard on appropriate dim
                placements.append(Shard(-1))
            elif axis_lower == "data":
                placements.append(Shard(0))
            else:
                placements.append(Replicate())
        else:
            placements.append(Replicate())

    return placements


def _placements_to_tensor_layout(placements: List[Placement], device_mesh) -> Tuple:
    """Convert DTensor placements back to TensorLayout axes.

    Args:
        placements: List of Placement objects.
        device_mesh: The DeviceMesh instance.

    Returns:
        Tuple of axis names for TensorLayout.
    """
    axes = []
    axis_names = device_mesh.axis_names

    for i, placement in enumerate(placements):
        if isinstance(placement, Replicate):
            axes.append(None)
        elif isinstance(placement, Shard):
            if i < len(axis_names):
                axes.append(axis_names[i])
            else:
                axes.append(f"dim_{i}")
        else:
            axes.append(None)

    return tuple(axes)


def distribute_data_input(
    per_process_batch: torch.Tensor, layout, batch_dim_name: str
) -> torch.Tensor:
    """Distribute input data with the corresponding layout.

    Note that the inputs here is a local worker batch. Within the local worker,
    the data needs to be further partitioned to map to each of the devices.

    Args:
        per_process_batch: Tensor that is already sharded to local process size.
        layout: TensorLayout for the distribution information.
        batch_dim_name: Name of the batch dimension axis.

    Returns:
        A global batch distributed according to layout.
    """
    # Avoid circular imports
    from keras.src.distribution import TensorLayout

    if _is_multi_process_mode():
        return per_process_batch

    if isinstance(layout, TensorLayout):
        device_mesh = layout.device_mesh.backend_mesh
        placements = _tensor_layout_to_placements(layout)
    else:
        device_mesh = layout.device_mesh.backend_mesh
        placements = _tensor_layout_to_placements(layout)

    return torch_distribute_tensor(per_process_batch, device_mesh, placements)


def initialize(
    job_addresses: Optional[str] = None,
    num_processes: Optional[int] = None,
    process_id: Optional[int] = None,
) -> None:
    """Initialize the distributed setting for PyTorch.

    This function initializes the PyTorch distributed process group
    for multi-device/multi-process training.

    Args:
        job_addresses: Comma-separated IP addresses for all jobs in the cluster.
        num_processes: Number of processes in the cluster.
        process_id: ID of the current process (0 to num_processes - 1).
    """
    global _distributed_initialized

    if _distributed_initialized:
        return

    # Check for environment variables
    if job_addresses is None and "KERAS_DISTRIBUTION_JOB_ADDRESSES" in os.environ:
        job_addresses = os.environ["KERAS_DISTRIBUTION_JOB_ADDRESSES"]
    if num_processes is None and "KERAS_DISTRIBUTION_NUM_PROCESSES" in os.environ:
        num_processes = int(os.environ["KERAS_DISTRIBUTION_NUM_PROCESSES"])
    if process_id is None and "KERAS_DISTRIBUTION_PROCESS_ID" in os.environ:
        process_id = int(os.environ["KERAS_DISTRIBUTION_PROCESS_ID"])

    # Set environment variables for PyTorch distributed
    if job_addresses:
        if "," in job_addresses:
            job_addresses_list = job_addresses.split(",")
            os.environ["MASTER_ADDR"] = job_addresses_list[0]
        else:
            os.environ["MASTER_ADDR"] = job_addresses
    else:
        os.environ.setdefault("MASTER_ADDR", "localhost")

    if process_id is not None:
        os.environ["MASTER_PORT"] = str(process_id + 29500)
    else:
        os.environ.setdefault("MASTER_PORT", "29500")

    if num_processes is not None:
        os.environ["WORLD_SIZE"] = str(num_processes)

    if process_id is not None:
        os.environ["RANK"] = str(process_id)

    # Initialize process group
    if not dist.is_initialized():
        # Use NCCL backend for GPU, GLOO for CPU
        if torch.cuda.is_available():
            backend = "nccl"
        else:
            backend = "gloo"

        dist.init_process_group(
            backend=backend,
            init_method="env://",
        )

    _distributed_initialized = True


def num_processes() -> int:
    """Return the number of processes for the current distribution setting."""
    if _is_multi_process_mode():
        return dist.get_world_size()
    return 1


def process_id() -> int:
    """Return the current process ID for the distribution setting."""
    if _is_multi_process_mode():
        return dist.get_rank()
    return 0


def _to_backend_device(device_name: str) -> torch.device:
    """Convert device name string to torch.device.

    Args:
        device_name: Device name like "cuda:0", "cpu", "gpu:1".

    Returns:
        torch.device instance.
    """
    if isinstance(device_name, torch.device):
        return device_name

    device_name = str(device_name).lower()
    if "gpu" in device_name:
        device_name = device_name.replace("gpu", "cuda")

    return torch.device(device_name)


def _to_backend_mesh(device_mesh) -> torch.distributed.device_mesh.DeviceMesh:
    """Convert the DeviceMesh to PyTorch backend specific DeviceMesh.

    Args:
        device_mesh: DeviceMesh instance to convert.

    Returns:
        A torch.distributed.device_mesh.DeviceMesh instance.
    """
    # Check cache first
    cache_key = (tuple(device_mesh.shape), tuple(device_mesh.axis_names))
    if cache_key in _device_mesh_cache:
        return _device_mesh_cache[cache_key]

    shape = device_mesh.shape
    axis_names = device_mesh.axis_names

    # In multi-process PyTorch, each process only has access to its local GPU(s)
    # We need to create a DeviceMesh that only includes the local devices
    # to avoid "Duplicate GPU detected" errors
    if _is_multi_process_mode() and torch.cuda.is_available():
        # Get the number of local GPUs for this process
        num_local_gpus = torch.cuda.device_count()

        # Get the axis name for data parallelism (usually first axis)
        if len(axis_names) > 0:
            data_axis_name = axis_names[0]
        else:
            data_axis_name = "batch"

        # Create a local mesh for this process
        # Each process only sees its local GPUs
        local_shape = (num_local_gpus,)

        # Create PyTorch DeviceMesh with only local devices
        # Note: In PyTorch DTensor, each rank only needs to know about its local devices
        torch_mesh = init_device_mesh(
            "cuda",
            local_shape,
            mesh_dim_names=(data_axis_name,),
        )

        _device_mesh_cache[cache_key] = torch_mesh
        return torch_mesh

    # For single-process or CPU case, use all devices
    # Convert device strings to actual devices
    devices = []
    for device_str in device_mesh.devices.flatten():
        device = _to_backend_device(device_str)
        devices.append(device)

    # Create PyTorch DeviceMesh
    torch_mesh = init_device_mesh(
        "cuda" if torch.cuda.is_available() else "cpu",
        shape,
        mesh_dim_names=axis_names,
    )

    _device_mesh_cache[cache_key] = torch_mesh
    return torch_mesh


def _to_backend_layout(tensor_layout) -> Tuple:
    """Convert the TensorLayout to DTensor placement specifications.

    Args:
        tensor_layout: TensorLayout instance to convert.

    Returns:
        A tuple of (device_mesh, placements) for DTensor operations.
    """
    if tensor_layout.device_mesh is None:
        raise ValueError(
            "Cannot create sharding when device mesh is not set "
            "for TensorLayout."
        )

    torch_mesh = _to_backend_mesh(tensor_layout.device_mesh)
    placements = _tensor_layout_to_placements(tensor_layout)

    return (torch_mesh, placements)


# Path Adapter for Keras/PyTorch Path Conversion

class TorchPathAdapter:
    """Adapt Keras path format to PyTorch format for regex matching.

    Keras uses / separators (e.g., 'dense/kernel')
    PyTorch uses . separators (e.g., 'dense.weight')

    This adapter allows the same regex patterns to work for both backends.
    """

    # Common parameter name mappings between Keras and PyTorch
    PARAM_NAME_MAPPINGS = {
        "kernel": ["weight", "kernel"],
        "bias": ["bias"],
        "gamma": ["weight"],
        "beta": ["bias"],
        "moving_mean": ["running_mean"],
        "moving_variance": ["running_var"],
    }

    # Reverse mapping for PyTorch to Keras
    REVERSE_MAPPINGS = {
        "weight": "kernel",
        "bias": "bias",
        "running_mean": "moving_mean",
        "running_var": "moving_variance",
    }

    @staticmethod
    def keras_to_torch(keras_path: str) -> str:
        """Convert Keras path to PyTorch format for matching.

        Args:
            keras_path: Path in Keras format, e.g., 'dense/kernel'

        Returns:
            Path in PyTorch regex format, e.g., 'dense\\..*weight'
        """
        # Split the path
        parts = keras_path.split("/")

        if len(parts) == 1:
            # Simple path like 'kernel'
            return TorchPathAdapter._convert_param_name(parts[0])

        # Convert layer name (first part)
        layer_name = parts[0]

        # Convert parameter name (last part)
        param_name = TorchPathAdapter._convert_param_name(parts[-1])

        # Handle wildcards
        if "*" in keras_path:
            # Replace * with regex pattern that matches both formats
            pattern = keras_path.replace(".", "\\.")
            for torch_name, keras_names in TorchPathAdapter.PARAM_NAME_MAPPINGS.items():
                for keras_name in keras_names:
                    if keras_name in pattern:
                        # Create regex that matches both
                        alternatives = "|".join(
                            TorchPathAdapter.PARAM_NAME_MAPPINGS.get(
                                torch_name, [torch_name]
                            )
                        )
                        pattern = pattern.replace(
                            f"*{keras_name}",
                            f"\\.({alternatives})"
                        )
                        break
            return pattern

        # Direct path conversion without wildcards
        if len(parts) >= 2:
            # Use the first alternative for direct mapping
            torch_param = TorchPathAdapter._convert_param_name(parts[-1])
            return f"{layer_name}.{torch_param}"

        return layer_name

    @staticmethod
    def _convert_param_name(param_name: str) -> str:
        """Convert a parameter name to PyTorch format.

        Args:
            param_name: Parameter name like 'kernel', 'bias'

        Returns:
            PyTorch parameter name like 'weight', 'bias'
        """
        # Check if it's a known parameter
        if param_name in TorchPathAdapter.REVERSE_MAPPINGS:
            return TorchPathAdapter.REVERSE_MAPPINGS[param_name]

        # Return original if no mapping
        return param_name

    @staticmethod
    def torch_to_keras(torch_path: str) -> str:
        """Convert PyTorch path to Keras format.

        Args:
            torch_path: Path in PyTorch format, e.g., 'dense.weight'

        Returns:
            Path in Keras format, e.g., 'dense/kernel'
        """
        # Convert dots back to slashes
        keras_path = torch_path.replace(".", "/")

        # Handle parameter name conversion
        for torch_name, keras_name in TorchPathAdapter.REVERSE_MAPPINGS.items():
            keras_path = keras_path.replace(torch_name, keras_name)

        return keras_path

    @staticmethod
    def matches_keras_pattern(keras_pattern: str, torch_path: str) -> bool:
        """Check if a PyTorch path matches a Keras regex pattern.

        Args:
            keras_pattern: Pattern in Keras format, e.g., 'dense.*kernel'
            torch_path: Path in PyTorch format, e.g., 'dense.weight'

        Returns:
            True if the path matches the pattern.
        """
        # Convert Keras pattern to PyTorch format
        torch_pattern = TorchPathAdapter.keras_to_torch(keras_pattern)

        # Try to match
        try:
            return re.search(torch_pattern, torch_path) is not None
        except re.error:
            # Fall back to direct comparison if regex is invalid
            return torch_pattern == torch_path


def get_sharded_tensor(
    tensor: torch.Tensor,
    device_mesh,
    placements: List[Placement],
) -> torch.Tensor:
    """Get a sharded DTensor from a regular tensor.

    This is a utility function for creating distributed tensors
    from regular tensors.

    Args:
        tensor: The input tensor to distribute.
        device_mesh: The DeviceMesh to distribute across.
        placements: List of placements specifying sharding.

    Returns:
        A DTensor distributed according to the placements.
    """
    return torch_distribute_tensor(tensor, device_mesh, placements)


def get_replicated_tensor(tensor: torch.Tensor, device_mesh) -> torch.Tensor:
    """Get a replicated DTensor from a regular tensor.

    Args:
        tensor: The input tensor to replicate.
        device_mesh: The DeviceMesh to replicate across.

    Returns:
        A replicated DTensor.
    """
    return torch_distribute_tensor(
        tensor, device_mesh, [Replicate()] * len(device_mesh.shape)
    )


# Model Parallelism Support

def infer_parallel_style(
    module: torch.nn.Module,
    param_name: str,
    sharding_spec: Tuple,
) -> Optional[str]:
    """Infer the parallel style from Keras sharding spec.

    Maps Keras sharding specs (tuples) to PyTorch Parallel Styles.
    Example: Linear Layer
    - Keras Dense Kernel: (None, 'model') -> Shard output features
    - PyTorch Linear Weight: (Out, In)

    Args:
        module: The PyTorch module containing the parameter.
        param_name: Name of the parameter (in PyTorch format).
        sharding_spec: Keras sharding specification tuple.

    Returns:
        Parallel style string: "colwise", "rowwise", or None.
    """
    # Identify the mesh axis used for model parallelism
    model_axis = None
    for i, axis in enumerate(sharding_spec):
        if axis == "model":
            model_axis = i
            break

    if model_axis is None:
        return None

    # Check if module is a Linear layer
    if isinstance(module, torch.nn.Linear):
        # Case A: Sharding the 2nd dim of the weight (Output Features)
        if model_axis == 1:
            return "colwise"

        # Case B: Sharding the 1st dim of the weight (Input Features)
        elif model_axis == 0:
            return "rowwise"

    # For Conv2D, similar logic applies to weight dimensions
    if isinstance(module, torch.nn.Conv2d):
        if model_axis == 0:
            return "colwise"
        elif model_axis == 1:
            return "rowwise"

    return None


def apply_tensor_parallelism(
    module: torch.nn.Module,
    layout_map: "LayoutMap",
    device_mesh: torch.distributed.device_mesh.DeviceMesh,
) -> torch.nn.Module:
    """Apply tensor parallelism to a module based on layout map.

    Args:
        module: The PyTorch module to parallelize.
        layout_map: Keras LayoutMap with sharding specifications.
        device_mesh: PyTorch DeviceMesh for distribution.

    Returns:
        The parallelized module.
    """
    # Import here to avoid circular imports
    from keras.src.distribution import LayoutMap, TensorLayout

    def parallelize_fn(name: str, module: torch.nn.Module) -> torch.nn.Module:
        """Inner function for parallelizing a module."""
        # Get all parameter paths for this module
        for param_name, param in module.named_parameters():
            # Construct the Keras-style path
            full_path = f"{name}.{param_name}" if name else param_name

            # Try to find matching layout
            layout = None
            for key in layout_map:
                if re.search(key, full_path):
                    layout = layout_map[key]
                    break

            if layout is not None:
                # Infer parallel style
                if isinstance(layout, TensorLayout):
                    sharding_spec = layout.axes
                else:
                    sharding_spec = layout

                parallel_style = infer_parallel_style(
                    module, param_name, sharding_spec
                )

                if parallel_style == "colwise":
                    parallel_style_fn = ColwiseParallel()
                elif parallel_style == "rowwise":
                    parallel_style_fn = RowwiseParallel()
                else:
                    continue

                # Apply parallelization
                try:
                    module = parallelize_module(
                        module,
                        device_mesh,
                        {param_name: parallel_style_fn}
                    )
                except Exception:
                    # Parallelization may fail for some modules
                    pass

        return module

    # Apply parallelization recursively
    return parallelize_fn("", module)


# Dataset Distribution Support

def distribute_dataset(
    dataset: torch.utils.data.Dataset,
    layout,
    batch_dim_name: str = "batch",
    num_replicas: Optional[int] = None,
    rank: Optional[int] = None,
) -> torch.utils.data.DataLoader:
    """Create a distributed data loader from a dataset.

    Args:
        dataset: The original dataset to distribute.
        layout: TensorLayout for the data distribution.
        batch_dim_name: Name of the batch dimension.
        num_replicas: Number of replicas (devices/processes).
        rank: Current process rank.

    Returns:
        A distributed DataLoader.
    """
    # Get number of replicas and rank
    if num_replicas is None:
        num_replicas = num_processes()
    if rank is None:
        rank = process_id()

    # Create distributed sampler
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=num_replicas,
        rank=rank,
        shuffle=True,
        drop_last=False,
    )

    # Determine batch size per replica
    batch_size = getattr(dataset, "batch_size", 32)
    batch_size_per_replica = max(1, batch_size // num_replicas)

    # Create data loader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size_per_replica,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return dataloader


def cleanup_distributed() -> None:
    """Clean up distributed resources."""
    global _distributed_initialized, _device_mesh_cache

    if dist.is_initialized():
        dist.destroy_process_group()

    _distributed_initialized = False
    _device_mesh_cache = {}
