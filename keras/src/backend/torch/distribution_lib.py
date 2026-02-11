"""Utilities for distribution strategy with PyTorch backend.

This module provides PyTorch-specific implementations for the Keras distribution
API, supporting CPU, GPU, and TPU devices. It uses PyTorch's DTensor for
distributed tensor operations and provides adapters for Keras-style layer
parameter naming conventions.
"""

import os
import re
from typing import Any, Optional, Tuple, Union, Dict, List

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
from torch.distributed.tensor import DTensor

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
    return dist.is_available() and dist.is_initialized()


def list_devices(device_type: Optional[str] = None) -> list:
    """Return all the available devices based on the device type.
    
    Args:
        device_type: One of "cpu", "gpu" or "tpu". Defaults to all available.
        
    Returns:
        List of device strings.
    """
    _check_torch_available()
    
    devices = []
    
    if device_type is None or device_type == "tpu":
        try:
            import torch_xla.core.xla_model as xm
            devices.extend([f"tpu:{i}" for i in range(xm.xrt_world_size())])
        except ImportError:
            pass
    
    if device_type is None or device_type == "gpu":
        if torch.cuda.is_available():
            devices.extend([f"cuda:{i}" for i in range(torch.cuda.device_count())])
    
    if device_type is None or device_type == "cpu":
        if not devices:  # Only add CPU devices if no accelerators found
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
        return torch.cuda.device_count() if torch.cuda.is_available() else 0
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
        device_name: Device name like "cpu:0", "cuda:1", "tpu:0"
        
    Returns:
        PyTorch device object.
    """
    device_name = device_name.lower()
    if device_name.startswith("tpu"):
        import torch_xla.core.xla_model as xm
        return xm.xla_device()
    elif device_name.startswith("cuda"):
        return torch.device(f"cuda:{device_name.split(':')[1]}")
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
    # Convert Keras path format to PyTorch format
    # e.g., "dense/kernel" -> "dense.weight", "dense/bias" -> "dense.bias"
    
    torch_path = keras_path.replace('/', '.')
    
    # Handle layer-specific parameter naming
    # Keras: layer_name/parameter_name -> PyTorch: layer_name.parameter_name
    # Common patterns:
    # - kernel -> weight
    # - bias -> bias  
    # - gamma -> weight (for normalization layers)
    # - beta -> bias (for normalization layers)
    # - running_mean -> running_mean
    # - running_var -> running_var
    
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
    # Convert PyTorch path format to Keras format
    # e.g., "dense.weight" -> "dense/kernel", "dense.bias" -> "dense/bias"
    
    replacements = [
        ('.weight', '.kernel'),  # Handle this first for Linear layers
        ('.weight', '/kernel'),  # For other layers
        ('.bias', '/bias'),
        ('.running_mean', '/moving_mean'),
        ('.running_var', '/moving_variance'),
    ]
    
    keras_path = torch_path
    
    # More specific replacements first
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


def _to_backend_mesh(device_mesh):
    """Convert Keras DeviceMesh to PyTorch DTensor DeviceMesh.
    
    Args:
        device_mesh: Keras DeviceMesh instance
        
    Returns:
        PyTorch DeviceMesh instance.
    """
    if device_mesh is None:
        return None
    
    # Convert device mesh to PyTorch format
    # Get the actual device list from the Keras mesh
    devices = device_mesh.devices.flatten().tolist()
    
    # Create PyTorch device mesh
    torch_mesh = init_device_mesh(
        backend=_get_torch_backend(devices),
        mesh_shape=device_mesh.shape,
        mesh_dim_names=device_mesh.axis_names
    )
    
    return torch_mesh


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


def _create_placement_from_layout(layout_axes, mesh_dim_names):
    """Create PyTorch DTensor placement from Keras layout axes.
    
    Args:
        layout_axes: Tuple of axis names (e.g., ('model', None))
        mesh_dim_names: Tuple of mesh dimension names (e.g., ('model',))
        
    Returns:
        List of placements (Shard or Replicate)
    """
    from torch.distributed.tensor import Replicate, Shard
    
    placements = []
    
    for axis_name in layout_axes:
        if axis_name is None:
            placements.append(Replicate())
        else:
            # Find the mesh dimension index for this axis
            if axis_name in mesh_dim_names:
                mesh_dim = mesh_dim_names.index(axis_name)
                placements.append(Shard(mesh_dim))
            else:
                placements.append(Replicate())
    
    return placements


# Global state for distribution
_distributed_initialized = False
_device_mesh_cache = {}


def initialize(job_addresses: Optional[str] = None, 
               num_processes: Optional[int] = None, 
               process_id: Optional[int] = None,
               backend: str = "nccl"):
    """Initialize the distribution system for PyTorch.
    
    Args:
        job_addresses: Comma separated IP addresses for all jobs
        num_processes: Number of worker processes
        process_id: Current worker process ID
        backend: Distribution backend ('nccl' for GPU, 'gloo' for CPU)
    """
    global _distributed_initialized
    
    _check_torch_available()
    
    if _distributed_initialized:
        return
    
    # Determine backend based on available devices
    if backend == "auto":
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            backend = "nccl"
        else:
            backend = "gloo"
    
    # Initialize torch distributed
    if num_processes is not None and num_processes > 1:
        torch.distributed.init_process_group(
            backend=backend,
            init_method="env://",
            world_size=num_processes,
            rank=process_id if process_id is not None else 0
        )
        
        # Set CUDA device for this process
        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            torch.cuda.set_device(local_rank)
    
    _distributed_initialized = True


def shutdown():
    """Shutdown the distribution system."""
    global _distributed_initialized, _device_mesh_cache
    
    if _distributed_initialized and dist.is_initialized():
        dist.destroy_process_group()
    
    _distributed_initialized = False
    _device_mesh_cache = {}


def distribute_variable(value, layout, device_mesh=None):
    """Create a distributed variable (DTensor) from a PyTorch tensor.
    
    This function implements ACTUAL physical sharding using PyTorch DTensor.
    
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
    
    # Check if distributed is initialized
    if not _check_distributed_initialized():
        # Return original tensor if not in distributed mode
        return value
    
    # Handle DTensor input
    if isinstance(value, DTensor):
        return value
    
    # Get mesh - use provided or create default
    if device_mesh is None:
        device_mesh = _get_default_device_mesh()
    
    if device_mesh is None:
        return value
    
    # Create placements from layout axes
    placements = _create_placement_from_layout(
        layout.axes, 
        device_mesh.mesh_dim_names
    )
    
    # Create DTensor from tensor
    # Use distribute_tensor to properly shard the tensor
    try:
        dtensor = tp.distribute_tensor(
            value.contiguous(),
            device_mesh,
            placements
        )
        return dtensor
    except Exception as e:
        # Fallback to original tensor if DTensor creation fails
        print(f"Warning: Could not create DTensor: {e}")
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
    
    # Check if distributed is initialized
    if not _check_distributed_initialized():
        return tensor
    
    # Handle DTensor input
    if isinstance(tensor, DTensor):
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
        dtensor = tp.distribute_tensor(
            tensor.contiguous(),
            device_mesh,
            placements
        )
        return dtensor
    except Exception as e:
        print(f"Warning: Could not distribute tensor: {e}")
        return tensor


def _get_default_device_mesh():
    """Get the default PyTorch DeviceMesh for the current process.
    
    Returns:
        PyTorch DeviceMesh instance or None.
    """
    global _device_mesh_cache
    
    if "default" in _device_mesh_cache:
        return _device_mesh_cache["default"]
    
    # Create default mesh based on available devices
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            try:
                mesh = init_device_mesh(
                    backend="cuda",
                    mesh_shape=(num_gpus,),
                    mesh_dim_names=("model",)
                )
                _device_mesh_cache["default"] = mesh
                return mesh
            except Exception:
                pass
    
    return None


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
        return per_process_batch
    
    # Get mesh
    if device_mesh is None:
        device_mesh = _get_default_device_mesh()
    
    if device_mesh is None:
        return per_process_batch
    
    # Create DTensor for input
    try:
        # Move tensor to correct device first
        if torch.cuda.is_available():
            per_process_batch = per_process_batch.cuda()
        
        # Create placements
        placements = _create_placement_from_layout(
            layout.axes,
            device_mesh.mesh_dim_names
        )
        
        # Create DTensor
        dtensor = tp.distribute_tensor(
            per_process_batch.contiguous(),
            device_mesh,
            placements
        )
        return dtensor
        
    except Exception as e:
        print(f"Warning: Could not distribute input data: {e}")
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
            self.mesh = init_device_mesh(
                backend=_get_torch_backend(list_devices()),
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

