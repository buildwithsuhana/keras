import torch
import torch.distributed as dist
import numpy as np
import re

# Use the system's preferred DeviceMesh import path
from torch.distributed.device_mesh import DeviceMesh as TorchDeviceMesh
from torch.distributed._tensor import DTensor, Replicate, Shard
from torch.distributed._tensor import distribute_tensor as torch_distribute_tensor

from keras.src.backend.torch.core import convert_to_tensor


def _is_dtensor(tensor):
    """Check if a tensor is a PyTorch DTensor."""
    return isinstance(tensor, DTensor)


def _ensure_dtensor(tensor, mesh, placements=None):
    """Convert a regular torch.Tensor to DTensor if needed.

    This function ensures that when operating with DTensors, all operands
    are DTensors. If the input is already a DTensor, it is returned as-is.
    If it's a regular tensor, it is converted to a DTensor with the given
    mesh and placements.

    Args:
        tensor: A torch.Tensor or DTensor
        mesh: The DeviceMesh to use for distribution
        placements: The placements for the DTensor. If None, uses Replicate.

    Returns:
        A DTensor representation of the input
    """
    if _is_dtensor(tensor):
        return tensor

    # Regular tensor that needs to be converted to DTensor
    if placements is None:
        placements = (Replicate(),)

    return torch_distribute_tensor(tensor, mesh, placements)


class KerasDTensor(DTensor):
    """A custom DTensor subclass that automatically handles mixed operations.

    When a KerasDTensor is used in an operation with a regular torch.Tensor,
    this class automatically converts the regular tensor to a DTensor with
    compatible placements before performing the operation. This ensures that
    PyTorch DTensor operations work correctly even when torch.compile/dynamo
    intercepts operations at the dispatch level.
    """

    @staticmethod
    def __torch_dispatch__(func, types, args, kwargs):
        """Intercept torch operations and handle mixed DTensor/tensor operations."""
        from torch.distributed._tensor import DTensor as DTensorClass

        # Convert all DTensor and regular tensor arguments to DTensors
        new_args = []
        dtensor_mesh = None
        dtensor_placements = None

        # Find if any argument is a DTensor and get its mesh/placements
        for arg in args:
            if isinstance(arg, DTensorClass):
                dtensor_mesh = arg.device_mesh
                dtensor_placements = arg.placements
                break

        # Convert all arguments
        for arg in args:
            if isinstance(arg, DTensorClass):
                new_args.append(arg)
            elif isinstance(arg, torch.Tensor) and dtensor_mesh is not None:
                # Convert regular tensor to DTensor with compatible placements
                converted = torch_distribute_tensor(
                    arg, dtensor_mesh, dtensor_placements
                )
                new_args.append(converted)
            else:
                new_args.append(arg)

        # Handle kwargs similarly
        new_kwargs = {}
        if kwargs:
            for key, value in kwargs.items():
                if isinstance(value, DTensorClass):
                    new_kwargs[key] = value
                elif isinstance(value, torch.Tensor) and dtensor_mesh is not None:
                    new_kwargs[key] = torch_distribute_tensor(
                        value, dtensor_mesh, dtensor_placements
                    )
                else:
                    new_kwargs[key] = value

        return func(*new_args, **new_kwargs)


def _wrap_as_keras_dtensor(tensor, mesh, placements):
    """Wrap a tensor as KerasDTensor for automatic mixed operation handling.

    Args:
        tensor: A torch.Tensor to wrap
        mesh: The DeviceMesh for distribution
        placements: The placements for the DTensor

    Returns:
        A KerasDTensor instance
    """
    # Create a local tensor and wrap it with KerasDTensor
    # We use DTensor's internal _local_tensor to store the actual tensor
    if not isinstance(tensor, torch.Tensor):
        tensor = convert_to_tensor(tensor)

    # Create the DTensor first
    dtensor = torch_distribute_tensor(tensor, mesh, placements)

    # Wrap it as KerasDTensor by creating a new instance
    # We need to preserve the _local_tensor from the existing DTensor
    result = KerasDTensor(
        dtensor._local_tensor,
        mesh,
        placements,
        requires_grad=tensor.requires_grad
    )
    return result


def list_devices(device_type=None):
    """List available local devices."""
    device_type = device_type or "cuda"
    if device_type == "cuda":
        return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    return ["cpu"]


def get_device_count(device_type=None):
    """Return the total number of devices in the cluster."""
    if dist.is_initialized():
        return dist.get_world_size()
    return torch.cuda.device_count()


def initialize(job_addresses=None, num_processes=None, process_id=None):
    """Initialize the distributed process group for Torch SPMD."""
    if not dist.is_initialized():
        # SPMD usually utilizes the NCCL backend for GPU coordination
        dist.init_process_group(backend="nccl")

    if torch.cuda.is_available():
        # Map the process to a specific local physical GPU based on rank
        local_rank = dist.get_rank() % torch.cuda.device_count()
        torch.cuda.set_device(local_rank)


def _to_backend_mesh(device_mesh):
    """Bridge for DeviceMesh.backend_mesh"""
    # Your framework signature: DeviceMesh(device_type, mesh, mesh_dim_names=...)
    # mesh can be a tuple (shape) or a list of ranks.

    if device_mesh.devices is not None:
        # Convert device names to integer device indices
        # device_mesh.devices can be np.array(['cuda:0', 'cuda:1']) or similar
        device_ids = []
        for d in device_mesh.devices:
            if isinstance(d, str):
                # Extract device index from string like 'cuda:0' -> 0
                res = re.search(r'\d+', d)
                device_ids.append(int(res.group()) if res else 0)
            else:
                device_ids.append(int(d))
    else:
        # Fallback to shape if devices is None
        device_ids = device_mesh.shape

    # Use positional arguments as required by PyTorch's DeviceMesh
    return TorchDeviceMesh(
        "cuda" if torch.cuda.is_available() else "cpu",
        device_ids,
        mesh_dim_names=device_mesh.axis_names
    )


def _to_backend_layout(layout):
    """Convert Keras Layout to Torch placements."""
    torch_mesh = layout.device_mesh.backend_mesh
    placements = []
    for axis in layout.axes:
        if axis is None:
            placements.append(Replicate())
        else:
            # Find the index of the axis name in the mesh dimension names
            dim_index = list(torch_mesh.mesh_dim_names).index(axis)
            placements.append(Shard(dim_index))
    return placements


def distribute_variable(variable, layout):
    """Intercept variable creation to shard it immediately.

    Args:
        variable: Either a Keras Variable object or a torch.Tensor
        layout: The TensorLayout specifying how to shard the variable

    Returns:
        A torch.nn.Parameter containing the sharded DTensor
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        print(f"[BACKEND] Sharding variable. Placements: {layout.axes}")

    # Handle both Keras Variable and torch.Tensor inputs
    if hasattr(variable, 'value'):
        # It's a Keras Variable - get the underlying tensor
        tensor_value = variable.value
    else:
        # It's already a torch.Tensor
        tensor_value = variable

    # Distribute the tensor (convert to DTensor)
    sharded_tensor = distribute_tensor(tensor_value, layout)

    # Return as Parameter for optimizer compatibility
    return torch.nn.Parameter(sharded_tensor)


def distribute_tensor(value, layout):
    """The core engine for Model Parallelism.
    Converts a standard torch.Tensor into a sharded KerasDTensor.
    """
    from keras.src.distribution.distribution_lib import TensorLayout

    if not isinstance(layout, TensorLayout):
        return value

    # layout.backend_layout and layout.device_mesh.backend_mesh 
    # trigger the _to_backend hooks automatically.
    placements = layout.backend_layout
    torch_mesh = layout.device_mesh.backend_mesh

    if not isinstance(value, torch.Tensor):
        value = convert_to_tensor(value)

    # Create the DTensor
    dtensor = torch_distribute_tensor(value, torch_mesh, placements)

    # Wrap it as KerasDTensor for automatic mixed operation handling
    # This ensures that when torch.compile/dynamo intercepts operations,
    # the __torch_dispatch__ will handle mixed DTensor/tensor operations
    result = KerasDTensor(
        dtensor._local_tensor,
        torch_mesh,
        placements,
        requires_grad=value.requires_grad
    )
    return result


def distribute_data_input(data, layout):
    """Shard input data batches."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    if hasattr(data, "shape"):
        print(f"[Rank {rank}] Sharding input batch. Local shape: {data.shape}")
    return distribute_tensor(data, layout)


def num_processes():
    """Return the total number of processes in the cluster."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def process_id():
    """Return the rank of the current process."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def device_id():
    """Return the local device ID."""
    if torch.cuda.is_available():
        return torch.cuda.current_device()
    return 0


def backend_num_processes():
    return num_processes()