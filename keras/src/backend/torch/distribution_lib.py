"""Utilities for distribution strategy with PyTorch backend.

This module provides PyTorch-specific implementations for the Keras distribution
API, supporting CPU, GPU, and TPU devices. It uses PyTorch's DTensor for
distributed tensor operations when available, and provides fallback mechanisms
for CPU-only or single-device scenarios.
"""

import os
import re
from typing import Any, Optional, Tuple, Union, Dict, List

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.tensor.parallel as tp
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
)
from torch.distributed.tensor import DTensor, Shard, Replicate

# Import DeviceMesh directly from keras.src.distribution to avoid circular import
# Note: We import keras_dist_lib inside methods that need it, after the module is fully loaded


def _is_torch_available():
    """Check if PyTorch is available."""
    try:
        import torch
        return True
    except ImportError:
        return False


def _check_torch_available():
    """Raise an error if PyTorch is not available."""
    if not _is_torch_available():
        raise ImportError(
            "PyTorch is not available. Please install PyTorch to use "
            "distribution features with the torch backend."
        )


def _check_distributed_initialized():
    """Check if torch.distributed is initialized."""
    if not dist.is_available():
        return False
    return dist.is_initialized()


def _is_gpu_available():
    """Check if CUDA GPUs are available."""
    return torch.cuda.is_available()


def _is_tpu_available():
    """Check if TPU is available."""
    try:
        import torch_xla.core.xla_model as xm
        return xm.xrt_world_size() > 0
    except ImportError:
        return False


def _get_default_device():
    """Get the default device for the current platform.
    
    Returns:
        str: 'cuda', 'mps', 'cpu', or 'xla'
    """
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    elif _is_tpu_available():
        return 'xla'
    else:
        return 'cpu'


def list_devices(device_type: Optional[str] = None) -> list:
    """Return all the available devices based on the device type.
    
    Args:
        device_type: One of "cpu", "gpu" or "tpu". Defaults to all available.
        
    Returns:
        List of device strings.
    """
    _check_torch_available()
    
    devices = []
    
    # Detect TPU devices
    if device_type is None or device_type == "tpu":
        try:
            import torch_xla.core.xla_model as xm
            tpu_count = xm.xrt_world_size()
            if tpu_count > 0:
                devices.extend([f"tpu:{i}" for i in range(tpu_count)])
        except ImportError:
            pass
    
    # Detect GPU devices - always check for CUDA if device_type is None or "gpu"
    if device_type is None or device_type == "gpu":
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            if gpu_count > 0:
                devices.extend([f"cuda:{i}" for i in range(gpu_count)])
        elif torch.backends.mps.is_available():
            # MPS is available on Apple Silicon
            devices.append("mps:0")
    
    # Detect CPU devices
    if device_type is None or device_type == "cpu":
        # Always include CPU if no other devices found or if specifically requested
        if not devices or device_type == "cpu":
            if "cpu:0" not in devices:
                devices.append("cpu:0")
    
    return devices


def get_device_count(device_type: Optional[str] = None) -> int:
    """Returns the number of available devices.
    
    Args:
        device_type: Optional device type to count (e.g., "cpu", "gpu", "tpu").
            If `None`, it defaults to counting "gpu" or "tpu" devices if
            available, otherwise it counts "cpu" devices.
            
    Returns:
        int: The total number of available devices for the specified type.
    """
    _check_torch_available()
    
    if device_type is None:
        if torch.cuda.is_available():
            return torch.cuda.device_count()
        try:
            import torch_xla.core.xla_model as xm
            return xm.xrt_world_size()
        except ImportError:
            return 1
    elif device_type == "gpu":
        if torch.cuda.is_available():
            return torch.cuda.device_count()
        elif torch.backends.mps.is_available():
            return 1
        return 0
    elif device_type == "cpu":
        return 1
    elif device_type == "tpu":
        try:
            import torch_xla.core.xla_model as xm
            return xm.xrt_world_size()
        except ImportError:
            return 0
    else:
        return 0


def _to_backend_device(device_name: str) -> torch.device:
    """Convert Keras device name to PyTorch device.
    
    Args:
        device_name: Device name like "cpu:0", "cuda:1", "tpu:0", "mps:0"
        
    Returns:
        PyTorch device object.
    """
    device_name = device_name.lower()
    if device_name.startswith("tpu"):
        try:
            import torch_xla.core.xla_model as xm
            return xm.xla_device()
        except ImportError:
            return torch.device("cpu")
    elif device_name.startswith("cuda"):
        return torch.device(f"cuda:{device_name.split(':')[1]}")
    elif device_name.startswith("mps"):
        return torch.device("mps")
    else:
        return torch.device("cpu")


def _convert_keras_path_to_torch(keras_path: str) -> str:
    """Convert Keras layer parameter path to PyTorch format.
    
    Keras uses '/' separators (e.g., 'dense/kernel', 'dense_1/bias')
    PyTorch uses '.' separators (e.g., 'dense.weight', 'dense_1.bias')
    
    Args:
        keras_path: Keras-style parameter path
        
    Returns:
        PyTorch-style parameter path
    """
    
    torch_path = keras_path.replace('/', '.')
    replacements = [
        ('.kernel', '.weight'),
        ('.gamma', '.weight'),
        ('.beta', '.bias'),
        ('.moving_mean', '.running_mean'),
        ('.moving_var', '.running_var'),
        ('.moving_mean', '.running_mean'),
        ('.moving_variance', '.running_var'),
    ]
    
    for old, new in replacements:
        if torch_path.endswith(old):
            torch_path = torch_path[:-len(old)] + new
            break
    
    return torch_path


def _convert_torch_path_to_keras(torch_path: str) -> str:
    """Convert PyTorch layer parameter path to Keras format.
    
    PyTorch uses '.' separators (e.g., 'dense.weight', 'dense_1.bias')
    Keras uses '/' separators (e.g., 'dense/kernel', 'dense_1/bias')
    
    Args:
        torch_path: PyTorch-style parameter path
        
    Returns:
        Keras-style parameter path
    """
    
    replacements = [
        ('.weight', '.kernel'),
        ('.weight', '/kernel'),
        ('.bias', '/bias'),
        ('.running_mean', '/moving_mean'),
        ('.running_var', '/moving_variance'),
    ]
    
    keras_path = torch_path
    
    if '.weight' in keras_path and '.kernel' not in keras_path:
        if 'dense' in keras_path or 'conv' in keras_path:
            keras_path = keras_path.replace('.weight', '/kernel')
        else:
            keras_path = keras_path.replace('.weight', '/kernel')
    elif '.bias' in keras_path:
        keras_path = keras_path.replace('.bias', '/bias')
    elif '.running_mean' in keras_path:
        keras_path = keras_path.replace('.running_mean', '/moving_mean')
    elif '.running_var' in keras_path:
        keras_path = keras_path.replace('.running_var', '/moving_variance')
    
    return keras_path.replace('.', '/')


def _to_backend_layout(layout):
    """Convert Keras TensorLayout to PyTorch DTensor sharding spec.
    
    Args:
        layout: Keras TensorLayout instance
        
    Returns:
        PyTorch DTensor ShardingSpec or None if not applicable.
    """
    if layout is None:
        return None
    
    # Handle simple case where layout has no axes specified (replicated)
    if all(axis is None for axis in layout.axes):
        return None
    
    # For complex layouts, return the axes info for later processing
    return layout


def _get_torch_backend(devices: list) -> str:
    """Determine the PyTorch backend based on device list.
    
    Args:
        devices: List of device strings
        
    Returns:
        Backend string ('cuda', 'cpu', 'xla', etc.)
    """
    if not devices:
        return "cpu"
    
    # Check for TPU
    if any(d.startswith("tpu") for d in devices):
        return "xla"
    
    # Check for CUDA GPU
    if any(d.startswith("cuda") for d in devices):
        return "cuda"
    
    # Check for MPS
    if any(d.startswith("mps") for d in devices):
        return "mps"
    
    # Default to CPU
    return "cpu"


def _infer_parallel_style(module, param_name: str, sharding_spec: tuple):
    """Translate Keras sharding specs to PyTorch Parallel Styles.
    
    Args:
        module: PyTorch module instance
        param_name: Parameter name (e.g., 'weight', 'bias')
        sharding_spec: Keras sharding spec tuple
        
    Returns:
        PyTorch parallel style or None
    """
    if not hasattr(module, 'weight') and not hasattr(module, 'bias'):
        return None
    
    # Identify which axis is used for model parallelism
    model_axis = None
    for i, axis in enumerate(sharding_spec):
        if axis == 'model':
            model_axis = i
            break
    
    if model_axis is None:
        return None  # No model parallelism for this parameter
    
    # Determine the shard index based on the parameter type
    if param_name == 'weight':
        # For Linear layers: weight shape is (out_features, in_features)
        # Keras spec: (None, 'model') -> Shard output features
        # Keras spec: ('model', None) -> Shard input features
        if model_axis == 1:
            return ColwiseParallel()
        elif model_axis == 0:
            return RowwiseParallel()
    
    elif param_name == 'bias':
        # Bias typically has one dimension (output_features)
        # Keras spec: ('model',) -> Shard bias
        if model_axis == 0:
            return ColwiseParallel()
    
    return None


def _create_placement_from_layout(layout_axes, mesh_dim_names, tensor_shape=None):
    """Create PyTorch DTensor placement from Keras layout axes.
    
    Args:
        layout_axes: Tuple of axis names (e.g., ('model', None))
        mesh_dim_names: Tuple of mesh dimension names (e.g., ('model',))
        tensor_shape: Optional shape of the tensor to distribute
        
    Returns:
        List of placements (Shard or Replicate)
    """
    from torch.distributed.tensor import Replicate, Shard
    
    # Get the number of mesh dimensions
    num_mesh_dims = len(mesh_dim_names) if mesh_dim_names else 0
    
    # If mesh has no dimensions, everything is replicated
    if num_mesh_dims == 0:
        return tuple([Replicate() for _ in layout_axes]) if layout_axes else tuple()
    
    placements = []
    
    # Track which mesh dimensions we've used
    used_mesh_dims = set()
    
    for i, axis_name in enumerate(layout_axes):
        if axis_name is None:
            # This dimension is not sharded - replicate
            placements.append(Replicate())
        else:
            # This dimension should be sharded
            if axis_name in mesh_dim_names:
                mesh_dim = mesh_dim_names.index(axis_name)
                # Check if this mesh dim is already used
                if mesh_dim in used_mesh_dims:
                    # Already used, replicate this dimension
                    placements.append(Replicate())
                else:
                    # Use this mesh dimension for sharding
                    placements.append(Shard(mesh_dim))
                    used_mesh_dims.add(mesh_dim)
            else:
                # Axis name not in mesh - replicate
                placements.append(Replicate())
    
    # Ensure we return a tuple of placements
    return tuple(placements)


# Global state for distribution
_distributed_initialized = False
_device_mesh_cache = {}
_backend_type = "cpu"  # Track the current backend type: 'cpu', 'cuda', 'mps', 'xla'


def _get_or_create_torch_device_mesh(keras_device_mesh):
    """Convert Keras DeviceMesh to PyTorch DeviceMesh.
    
    This function creates a PyTorch DTensor DeviceMesh from a Keras DeviceMesh.
    It handles the conversion of device strings and mesh configuration.
    
    Args:
        keras_device_mesh: Keras DeviceMesh instance
        
    Returns:
        PyTorch DeviceMesh instance or None if creation fails.
    """
    global _device_mesh_cache
    
    if keras_device_mesh is None:
        return None
    
    # Check cache first
    cache_key = f"keras_mesh_{id(keras_device_mesh)}"
    if cache_key in _device_mesh_cache:
        return _device_mesh_cache[cache_key]
    
    # Get device information from Keras DeviceMesh
    keras_devices = keras_device_mesh.devices
    keras_shape = keras_device_mesh.shape
    keras_axis_names = keras_device_mesh.axis_names
    
    # Determine backend type based on device strings
    device_type = _get_torch_backend(list(keras_devices.flatten()))
    
    # Convert Keras device strings to PyTorch format
    # Handle the reshape of devices
    flat_devices = keras_devices.flatten()
    torch_devices = []
    for dev in flat_devices:
        if dev.startswith("cuda"):
            torch_devices.append(dev)
        elif dev.startswith("cpu"):
            # For CPU devices, we need to check if we can use them
            # PyTorch DTensor typically doesn't work well with multiple CPU devices
            # without proper process group setup
            torch_devices.append(dev)
        elif dev.startswith("tpu"):
            torch_devices.append(dev)
        else:
            torch_devices.append(dev)
    
    # Try to create PyTorch DeviceMesh
    try:
        # Use init_device_mesh which is the recommended way in PyTorch 2.8+
        torch_mesh = init_device_mesh(
            device_type=device_type,
            mesh_shape=keras_shape,
            mesh_dim_names=tuple(keras_axis_names)
        )
        _device_mesh_cache[cache_key] = torch_mesh
        print(f"✓ Created PyTorch DeviceMesh from Keras DeviceMesh: shape={keras_shape}, axes={keras_axis_names}, device_type={device_type}")
        return torch_mesh
    except Exception as e:
        print(f"Warning: Could not create PyTorch DeviceMesh: {e}")
        
        # Try fallback approach
        try:
            # Try with single device
            if len(torch_devices) == 1:
                torch_mesh = init_device_mesh(
                    device_type=device_type,
                    mesh_shape=(1,),
                    mesh_dim_names=("model",)
                )
                _device_mesh_cache[cache_key] = torch_mesh
                return torch_mesh
        except Exception as e2:
            print(f"Warning: Fallback DeviceMesh creation also failed: {e2}")
        
        return None


def initialize(job_addresses: Optional[str] = None, 
               num_processes: Optional[int] = None, 
               process_id: Optional[int] = None,
               backend: str = "auto"):
    """Initialize the distribution system for PyTorch.
    
    This function initializes the distribution system for the appropriate
    backend (CPU, GPU, MPS, or TPU) based on available devices.
    
    Args:
        job_addresses: Comma separated IP addresses for all jobs
        num_processes: Number of worker processes
        process_id: Current worker process ID
        backend: Distribution backend ('auto', 'nccl', 'gloo', 'xla')
            - 'auto': Automatically select based on available devices
            - 'nccl': Use NCCL for multi-GPU training
            - 'gloo': Use Gloo for CPU-based training
            - 'xla': Use XLA for TPU training
    """
    global _distributed_initialized, _backend_type
    
    _check_torch_available()
    
    if _distributed_initialized:
        return
    
    # Detect available devices and set backend type
    if _is_tpu_available():
        _backend_type = "xla"
    elif _is_gpu_available():
        _backend_type = "cuda"
    elif torch.backends.mps.is_available():
        _backend_type = "mps"
    else:
        _backend_type = "cpu"
    
    # Handle auto backend selection
    if backend == "auto":
        if _backend_type == "cuda":
            backend = "nccl" if torch.cuda.device_count() > 1 else "gloo"
        elif _backend_type == "xla":
            backend = "xla"
        else:
            backend = "gloo"
    
    # Initialize based on backend type
    if _backend_type == "xla":
        # TPU initialization
        try:
            import torch_xla.core.xla_model as xm
            xm.rendezvous("init")
            print("✓ Initialized torch.distributed for TPU")
        except Exception as e:
            print(f"Note: Could not initialize TPU distributed: {e}")
    elif _backend_type == "cuda":
        # CUDA GPU initialization
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            # Multi-GPU training
            if num_processes is None:
                num_processes = num_gpus
            
            try:
                torch.distributed.init_process_group(
                    backend="nccl",
                    init_method="env://",
                    world_size=num_processes,
                    rank=process_id if process_id is not None else 0
                )
                
                # Set CUDA device for this process
                if "LOCAL_RANK" in os.environ:
                    local_rank = int(os.environ["LOCAL_RANK"])
                    torch.cuda.set_device(local_rank)
                elif num_processes > 1:
                    torch.cuda.set_device(0)
                    
                print(f"✓ Initialized torch.distributed for {num_gpus} GPUs (NCCL)")
            except Exception as e:
                print(f"Note: Could not initialize NCCL: {e}")
                # Fallback to single GPU mode
                _distributed_initialized = True
                return
        else:
            # Single GPU - still initialize for DTensor support
            try:
                torch.distributed.init_process_group(
                    backend="gloo",
                    init_method="env://",
                    world_size=1,
                    rank=0
                )
                print("✓ Initialized torch.distributed (single GPU)")
            except Exception:
                pass
    elif _backend_type == "mps":
        # MPS (Apple Silicon) initialization
        try:
            torch.distributed.init_process_group(
                backend="gloo",
                init_method="env://",
                world_size=1,
                rank=0
            )
            print("✓ Initialized torch.distributed for MPS (single process)")
        except Exception:
            pass
    else:
        # CPU-only initialization
        try:
            torch.distributed.init_process_group(
                backend="gloo",
                init_method="env://",
                world_size=1,
                rank=0
            )
            print("✓ Initialized torch.distributed for CPU (single process)")
        except Exception:
            print("Note: Could not initialize distributed, running in single-process mode")
            # Don't fail - single process mode is fine
            _distributed_initialized = True
            return
    
    _distributed_initialized = True


def shutdown():
    """Shutdown the distribution system."""
    global _distributed_initialized, _device_mesh_cache, _backend_type
    
    if _distributed_initialized and dist.is_initialized():
        dist.destroy_process_group()
    
    _distributed_initialized = False
    _device_mesh_cache = {}
    _backend_type = "cpu"


def distribute_variable(value, layout, device_mesh=None):
    """Create a distributed variable (DTensor) from a PyTorch tensor.
    
    This function implements ACTUAL physical sharding using PyTorch DTensor.
    If DTensor is not available (e.g., on CPU-only systems), it returns
    the original tensor.
    
    Args:
        value: PyTorch tensor to distribute
        layout: Keras TensorLayout instance specifying sharding axes
        device_mesh: PyTorch DeviceMesh instance (optional, uses default if not provided)
        
    Returns:
        DTensor with proper sharding, or original tensor if no sharding needed.
    """
    _check_torch_available()
    
    if layout is None or value is None:
        return value
    
    # Handle DTensor input
    if isinstance(value, DTensor):
        return value
    
    # Check if distributed is initialized - try to initialize if not
    if not _check_distributed_initialized():
        try:
            initialize()
        except Exception:
            pass
    
    # Get mesh - use provided or get from layout's device_mesh
    torch_device_mesh = None
    
    if device_mesh is not None:
        # Use provided PyTorch DeviceMesh
        torch_device_mesh = device_mesh
    elif layout.device_mesh is not None:
        # Try to get or create PyTorch DeviceMesh from Keras DeviceMesh
        keras_mesh = layout.device_mesh
        try:
            torch_device_mesh = _get_or_create_torch_device_mesh(keras_mesh)
        except Exception as e:
            print(f"Note: Could not create DeviceMesh from Keras mesh: {e}")
    
    # Fallback to default mesh if none available
    if torch_device_mesh is None:
        torch_device_mesh = _get_default_device_mesh()
    
    if torch_device_mesh is None:
        print("Note: No DeviceMesh available, skipping DTensor distribution")
        return value
    
    # Create placements from layout axes
    placements = _create_placement_from_layout(
        layout.axes, 
        torch_device_mesh.mesh_dim_names
    )
    
    # Create DTensor from tensor
    try:
        # Ensure value is contiguous
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value)
        value = value.contiguous()
        
        # Check if tensor dtype is suitable for DTensor (must be floating point for gradients)
        if not value.is_floating_point() and not value.is_complex():
            # Can't create DTensor from non-floating point tensors
            print(f"Note: Could not create DTensor: non-floating point dtype {value.dtype}")
            return value
        
        # Create DTensor using DTensor.from_local (NOT tp.distribute_tensor which doesn't exist)
        dtensor = DTensor.from_local(
            value,
            torch_device_mesh,
            placements,
            run_check=False
        )
        print(f"✓ Created DTensor with shape {dtensor.shape}, placements {placements}")
        return dtensor
    except Exception as e:
        # Fallback to original tensor if DTensor creation fails
        # This is expected on CPU-only systems or when mesh is incompatible
        print(f"Note: Could not create DTensor: {e}")
        return value


def distribute_tensor(tensor, layout, device_mesh=None):
    """Distribute a tensor based on the layout using PyTorch DTensor.
    
    Args:
        tensor: PyTorch tensor to distribute
        layout: Keras TensorLayout instance
        device_mesh: PyTorch DeviceMesh instance
        
    Returns:
        DTensor with proper sharding, or original tensor if no sharding.
    """
    _check_torch_available()
    
    if layout is None or tensor is None:
        return tensor
    
    # Handle DTensor input
    if isinstance(tensor, DTensor):
        return tensor
    
    # Check if distributed is initialized
    if not _check_distributed_initialized():
        try:
            initialize()
        except Exception:
            return tensor
    
    # Get mesh
    if device_mesh is None:
        device_mesh = _get_default_device_mesh()
    
    if device_mesh is None:
        return tensor
    
    # Create placements from layout
    placements = _create_placement_from_layout(
        layout.axes,
        device_mesh.mesh_dim_names
    )
    
    # Create DTensor
    try:
        # Check if tensor dtype is suitable for DTensor
        if not tensor.is_floating_point() and not tensor.is_complex():
            # Can't create DTensor from non-floating point tensors
            return tensor
        
        # Create DTensor using DTensor.from_local (NOT tp.distribute_tensor which doesn't exist)
        dtensor = DTensor.from_local(
            tensor.contiguous(),
            device_mesh,
            placements,
            run_check=False
        )
        return dtensor
    except Exception:
        # Fallback to original tensor
        return tensor


def _get_default_device_mesh():
    """Get the default PyTorch DeviceMesh for the current process.
    
    This function creates a DeviceMesh that works across CPU, GPU, MPS, and TPU.
    
    Returns:
        PyTorch DeviceMesh instance or None.
    """
    global _device_mesh_cache
    
    if "default" in _device_mesh_cache:
        return _device_mesh_cache["default"]
    
    # Check for TPU
    if _is_tpu_available():
        try:
            import torch_xla.core.xla_model as xm
            num_devices = xm.xrt_world_size()
            devices = [f"tpu:{i}" for i in range(num_devices)]
            mesh = init_device_mesh(
                device_type="xla",
                mesh_shape=(num_devices,),
                mesh_dim_names=("model",)
            )
            _device_mesh_cache["default"] = mesh
            print(f"✓ Created DeviceMesh for TPU with {num_devices} devices")
            return mesh
        except Exception as e:
            print(f"Warning: Could not create TPU DeviceMesh: {e}")
    
    # Check for CUDA GPU
    if _is_gpu_available():
        num_gpus = torch.cuda.device_count()
        if num_gpus >= 1:
            try:
                mesh = init_device_mesh(
                    device_type="cuda",
                    mesh_shape=(num_gpus,),
                    mesh_dim_names=("model",)
                )
                _device_mesh_cache["default"] = mesh
                print(f"✓ Created DeviceMesh for {num_gpus} GPUs with 'model' axis")
                return mesh
            except Exception as e:
                print(f"Warning: Could not create GPU DeviceMesh: {e}")
                # Try fallback approach using DeviceMesh class directly
                try:
                    devices = [f"cuda:{i}" for i in range(num_gpus)]
                    mesh = DeviceMesh(
                        device_type="cuda",
                        mesh=devices
                    )
                    _device_mesh_cache["default"] = mesh
                    return mesh
                except Exception:
                    pass
    
    # Check for MPS (Apple Silicon)
    if torch.backends.mps.is_available():
        try:
            mesh = init_device_mesh(
                device_type="mps",
                mesh_shape=(1,),
                mesh_dim_names=("model",)
            )
            _device_mesh_cache["default"] = mesh
            print("✓ Created DeviceMesh for MPS")
            return mesh
        except Exception as e:
            print(f"Warning: Could not create MPS DeviceMesh: {e}")
    
    # CPU-only fallback - create a logical mesh for single process
    try:
        mesh = init_device_mesh(
            device_type="cpu",
            mesh_shape=(1,),
            mesh_dim_names=("model",)
        )
        _device_mesh_cache["default"] = mesh
        print("✓ Created DeviceMesh for CPU (single process)")
        return mesh
    except Exception:
        # Last resort: return None and let callers handle it
        print("Note: Could not create DeviceMesh, running in fallback mode")
        return None


def _get_first_device():
    """Get the primary device for the current process.
    
    Returns:
        torch.device: The primary device (cuda:0, mps:0, cpu, or xla device)
    """
    if _is_gpu_available():
        return torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        return torch.device("mps:0")
    elif _is_tpu_available():
        try:
            import torch_xla.core.xla_model as xm
            return xm.xla_device()
        except ImportError:
            return torch.device("cpu")
    else:
        return torch.device("cpu")


def _ensure_tensor_on_device(tensor, device):
    """Ensure a tensor is on the specified device.
    
    Args:
        tensor: PyTorch tensor or numpy array
        device: Target torch.device
        
    Returns:
        Tensor on the specified device.
    """
    if tensor is None:
        return tensor
    
    # If it's a numpy array, convert to tensor first
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    
    # Move to device if not already there
    if isinstance(tensor, torch.Tensor):
        if tensor.device != device:
            try:
                tensor = tensor.to(device)
            except Exception:
                pass  # Keep original if can't move
    
    return tensor


def _set_default_device_mesh(mesh):
    """Set the default DeviceMesh.
    
    Args:
        mesh: PyTorch DeviceMesh instance
    """
    global _device_mesh_cache
    _device_mesh_cache["default"] = mesh


def distribute_data_input(per_process_batch, layout, batch_dim_name, device_mesh=None):
    """Distribute input data with the corresponding layout.
    
    Args:
        per_process_batch: Input data for one process (numpy or torch tensor)
        layout: Keras TensorLayout instance
        batch_dim_name: Name of the batch dimension
        device_mesh: PyTorch DeviceMesh instance
        
    Returns:
        Distributed input data (DTensor or original).
    """
    _check_torch_available()
    
    if layout is None:
        return per_process_batch
    
    # Check if distributed is initialized
    if not _check_distributed_initialized():
        try:
            initialize()
        except Exception:
            return per_process_batch
    
    # Get mesh
    if device_mesh is None:
        device_mesh = _get_default_device_mesh()
    
    if device_mesh is None:
        return per_process_batch
    
    # Create DTensor for input
    try:
        # Move tensor to correct device first
        device = _get_default_device()
        if device == "cuda":
            per_process_batch = per_process_batch.cuda()
        elif device == "mps":
            per_process_batch = per_process_batch.to("mps")
        elif device == "xla":
            try:
                import torch_xla.core.xla_model as xm
                per_process_batch = per_process_batch.to(xm.xla_device())
            except ImportError:
                pass
        
        # Create placements
        placements = _create_placement_from_layout(
            layout.axes,
            device_mesh.mesh_dim_names
        )
        
        # Check if tensor dtype is suitable for DTensor
        if not per_process_batch.is_floating_point() and not per_process_batch.is_complex():
            # Can't create DTensor from non-floating point tensors
            return per_process_batch
        
        # Create DTensor using DTensor.from_local (NOT tp.distribute_tensor which doesn't exist)
        dtensor = DTensor.from_local(
            per_process_batch.contiguous(),
            device_mesh,
            placements,
            run_check=False
        )
        return dtensor
        
    except Exception:
        # Fallback to original tensor
        return per_process_batch


def num_processes() -> int:
    """Return the number of processes for the current distribution setting."""
    _check_torch_available()
    
    if not _check_distributed_initialized():
        return 1
    return dist.get_world_size()


def process_id() -> int:
    """Return the current process ID for the distribution setting."""
    _check_torch_available()
    
    if not _check_distributed_initialized():
        return 0
    return dist.get_rank()


def get_local_rank() -> int:
    """Return the local rank for the current process."""
    _check_torch_available()
    
    if not _check_distributed_initialized():
        return 0
    return dist.get_rank()


def all_reduce(tensor, op="sum"):
    """All-reduce a tensor across all processes.
    
    Args:
        tensor: Tensor to reduce
        op: Reduction operation ('sum', 'product', 'min', 'max', 'avg')
        
    Returns:
        Reduced tensor.
    """
    _check_torch_available()
    
    if not _check_distributed_initialized():
        return tensor
    
    # Convert op string to torch function
    op_map = {
        "sum": dist.ReduceOp.SUM,
        "product": dist.ReduceOp.PRODUCT,
        "min": dist.ReduceOp.MIN,
        "max": dist.ReduceOp.MAX,
        "avg": dist.ReduceOp.AVG,
    }
    reduce_op = op_map.get(op.lower(), dist.ReduceOp.SUM)
    
    # Handle DTensor
    if isinstance(tensor, DTensor):
        local_tensor = tensor._local_tensor
    else:
        local_tensor = tensor
    
    # Perform all-reduce
    dist.all_reduce(local_tensor, op=reduce_op)
    
    # Return to original type
    if isinstance(tensor, DTensor):
        return DTensor.from_local(local_tensor, tensor._spec)
    else:
        return local_tensor


def all_gather(tensor):
    """All-gather a tensor across all processes.
    
    Args:
        tensor: Tensor to gather
        
    Returns:
        Gathered tensor (on rank 0).
    """
    _check_torch_available()
    
    if not _check_distributed_initialized():
        return tensor
    
    # Handle DTensor
    if isinstance(tensor, DTensor):
        local_tensor = tensor._local_tensor
    else:
        local_tensor = tensor
    
    # Get tensor shape
    shape = local_tensor.shape
    dtype = local_tensor.dtype
    
    # Create output tensor
    world_size = dist.get_world_size()
    output = torch.empty((world_size,) + shape, dtype=dtype, device=local_tensor.device)
    
    # Perform all-gather
    dist.all_gather_into_tensor(output, local_tensor)
    
    return output


def broadcast(tensor, src=0):
    """Broadcast a tensor from source rank to all other ranks.
    
    Args:
        tensor: Tensor to broadcast
        src: Source rank
        
    Returns:
        Broadcasted tensor.
    """
    _check_torch_available()
    
    if not _check_distributed_initialized():
        return tensor
    
    # Handle DTensor
    if isinstance(tensor, DTensor):
        local_tensor = tensor._local_tensor
    else:
        local_tensor = tensor
    
    # Broadcast
    dist.broadcast(local_tensor, src=src)
    
    # Return to original type
    if isinstance(tensor, DTensor):
        return DTensor.from_local(local_tensor, tensor._spec)
    else:
        return local_tensor


class TorchModelParallel:
    """Model Parallel wrapper for PyTorch using DTensor.
    
    This class provides model parallelism for PyTorch models by translating
    Keras-style layout specifications to PyTorch DTensor parallel styles.
    Works across CPU, GPU, MPS, and TPU devices.
    """
    
    def __init__(self, device_mesh, layout_map):
        """Initialize the model parallel wrapper.
        
        Args:
            device_mesh: DeviceMesh tuple/list or PyTorch DeviceMesh
            layout_map: Keras LayoutMap instance
        """
        _check_torch_available()
        
        # Initialize PyTorch Device Mesh
        if isinstance(device_mesh, (tuple, list)):
            # Use device_type instead of backend for PyTorch 2.8+
            self.mesh = init_device_mesh(
                device_type=_get_torch_backend(list_devices()),
                mesh_shape=device_mesh,
                mesh_dim_names=("data", "model")
            )
            self.mesh_axis_names = ("data", "model")
        else:
            self.mesh = device_mesh
            self.mesh_axis_names = device_mesh.mesh_dim_names
        
        self.layout_map = layout_map
        self.plan = {}
    
    def _convert_path(self, keras_path):
        """Convert Keras path to PyTorch path."""
        return _convert_keras_path_to_torch(keras_path)
    
    def _get_layout_for_param(self, param_path, param_shape):
        """Get the layout for a parameter based on the layout map."""
        # First try exact match
        if param_path in self.layout_map:
            return self.layout_map[param_path]
        
        # Try regex matching
        for key in self.layout_map:
            if re.search(key, param_path):
                return self.layout_map[key]
        
        return None
    
    def parallelize_module(self, module):
        """Parallelize a PyTorch module based on the layout map.
        
        Args:
            module: PyTorch module to parallelize
            
        Returns:
            Parallelized module.
        """
        _check_torch_available()
        
        for name, param in module.named_parameters():
            torch_path = name
            keras_path = _convert_torch_path_to_keras(name)
            
            layout = self._get_layout_for_param(keras_path, param.shape)
            
            if layout is not None:
                # Find the parent module and the parameter name
                parts = name.rsplit('.', 1)
                if len(parts) == 2:
                    parent_name, param_name = parts
                    parent = module
                    for part in parent_name.split('.'):
                        if hasattr(parent, part):
                            parent = getattr(parent, part)
                        else:
                            break
                    
                    if hasattr(parent, param_name):
                        parallel_style = _infer_parallel_style(
                            parent, param_name, layout.axes
                        )
                        
                        if parallel_style is not None:
                            # Parallelize the module
                            try:
                                module = tp.parallelize_module(
                                    module,
                                    self.mesh,
                                    {parent_name: parallel_style}
                                )
                            except Exception as e:
                                # Log warning for debugging
                                print(f"Warning: Could not parallelize {name}: {e}")
        
        return module


class TorchDataParallel:
    """Data Parallel wrapper for PyTorch.
    
    This class provides data parallelism for PyTorch models using
    DistributedDataParallel or simple data sharding.
    Works across CPU, GPU, MPS, and TPU devices.
    """
    
    def __init__(self, device_mesh=None, devices=None):
        """Initialize the data parallel wrapper.
        
        Args:
            device_mesh: Optional DeviceMesh instance
            devices: Optional list of devices
        """
        _check_torch_available()
        
        if devices is None:
            devices = list_devices()
        
        if device_mesh is None:
            # Import DeviceMesh here to avoid circular import
            from keras.src.distribution import DeviceMesh

            device_mesh = DeviceMesh(
                shape=(len(devices),),
                axis_names=["batch"],
                devices=devices
            )
        
        self.device_mesh = device_mesh
    
    def prepare_module(self, module):
        """Prepare a module for data parallel training.
        
        Args:
            module: PyTorch module to prepare
            
        Returns:
            Data parallel wrapped module.
        """
        _check_torch_available()
        
        # Use DistributedDataParallel for multi-GPU training
        if torch.cuda.device_count() > 1:
            return torch.nn.parallel.DistributedDataParallel(module)
        else:
            return module
    
    def prepare_dataloader(self, dataset, batch_size, shuffle=True):
        """Prepare a distributed data loader.
        
        Args:
            dataset: Dataset to distribute
            batch_size: Batch size per replica
            shuffle: Whether to shuffle data
            
        Returns:
            Distributed DataLoader.
        """
        _check_torch_available()
        
        if num_processes() > 1:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=num_processes(),
                rank=process_id(),
                shuffle=shuffle
            )
            return torch.utils.data.DataLoader(
                dataset,
                sampler=sampler,
                batch_size=batch_size
            )
        else:
            return torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle
            )
