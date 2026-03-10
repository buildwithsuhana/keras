"""Utilities for distribution strategy with Torch backend."""

import os
import re
import torch
import torch.nn as nn
import numpy as np
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor as torch_distribute_tensor
from torch.distributed.tensor import Shard, Replicate
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)

from keras.src.backend.common import global_state

DEBUG = os.environ.get("KERAS_DISTRIBUTION_DEBUG", "0") == "1"

from torch.distributed.tensor._sharding_prop import ShardingPropagator

def register_fallback_sharding_strategy():
    """Register a fallback sharding strategy for unsupported operators."""
    def fallback_sharding_strategy(op_info):
        if DEBUG:
            print(f"[DEBUG] Operator {op_info.schema} does not have a registered sharding strategy. Falling back to replication.")
        return None  # Returning None disables sharding for this operator

    if hasattr(ShardingPropagator, "register_fallback"):
        ShardingPropagator.register_fallback(fallback_sharding_strategy)
    else:
        if DEBUG:
            print("[DEBUG] ShardingPropagator does not support fallback registration. Skipping fallback setup.")

# Call the fallback registration during initialization
register_fallback_sharding_strategy()


def list_devices(device_type=None):
    """Return all the available devices based on the device type."""
    if device_type is None:
        if torch.cuda.is_available():
            device_type = "cuda"
        else:
            device_type = "cpu"
    else:
        device_type = device_type.lower()
        if device_type == "gpu":
            device_type = "cuda"

    if device_type == "cuda":
        num_devices = torch.cuda.device_count()
    elif device_type == "cpu":
        num_devices = 1
    else:
        num_devices = 0

    return [f"{device_type}:{i}" for i in range(num_devices)]


def get_device_count(device_type=None):
    """Returns the number of available Torch devices."""
    return len(list_devices(device_type))


def initialize(job_addresses=None, num_processes=None, process_id=None):
    """Initializes the distribution system for multi-host/process setting."""
    if not torch.distributed.is_initialized():
        import os

        if job_addresses:
            master_addr, master_port = job_addresses.split(",")[0].split(":")
            os.environ["MASTER_ADDR"] = master_addr
            os.environ["MASTER_PORT"] = master_port
        
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "127.0.0.1"
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29500"

        if num_processes is not None:
            os.environ["WORLD_SIZE"] = str(num_processes)
        if "WORLD_SIZE" not in os.environ:
            os.environ["WORLD_SIZE"] = "1"
            
        if process_id is not None:
            os.environ["RANK"] = str(process_id)
        if "RANK" not in os.environ:
            os.environ["RANK"] = "0"

        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if DEBUG:
            print(f"[DEBUG] Initializing torch distributed with {backend} backend")
            
        torch.distributed.init_process_group(backend=backend)


def num_processes():
    """Return the number of processes for the current distribution setting."""
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def process_id():
    """Return the current process ID for the distribution setting."""
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def _to_backend_mesh(device_mesh):
    """Convert the DeviceMesh to Torch backend specific Mesh."""
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    mesh_shape = device_mesh.shape
    if DEBUG:
        print(f"[DEBUG] Creating device mesh: type={device_type}, shape={mesh_shape}, axis_names={device_mesh.axis_names}")
    return init_device_mesh(
        device_type, mesh_shape, mesh_dim_names=device_mesh.axis_names
    )


def _to_backend_layout(tensor_layout):
    """Convert the TensorLayout to Torch backend specific Sharding."""
    if tensor_layout.device_mesh is None:
        raise ValueError(
            "Cannot create sharding when device mesh is not set for "
            "TensorLayout."
        )

    device_mesh = tensor_layout.device_mesh
    torch_mesh = device_mesh.backend_mesh

    placements = []
    for mesh_dim_name in device_mesh.axis_names:
        shard_dim = None
        for i, axis in enumerate(tensor_layout.axes):
            if axis == mesh_dim_name:
                shard_dim = i
                break
        if shard_dim is not None:
            placements.append(Shard(shard_dim))
        else:
            placements.append(Replicate())
        
    return (torch_mesh, placements)


def distribute_variable(value, layout):
    """Distribute the variable based on the layout."""
    is_parameter = isinstance(value, torch.nn.Parameter)
    requires_grad = value.requires_grad if is_parameter else False
    
    sharded_tensor = distribute_tensor(value, layout)
    
    if is_parameter:
        res = torch.nn.Parameter(sharded_tensor, requires_grad=requires_grad)
        if hasattr(value, "constraint"):
            res.constraint = value.constraint
        else:
            res.constraint = None
        return res
    return sharded_tensor


def distribute_tensor(tensor, layout):
    """Distribute the tensor based on the layout."""
    from keras.src.distribution import TensorLayout

    if isinstance(layout, TensorLayout):
        layout = layout.backend_layout

    mesh, placements = layout
    mesh_device_type = mesh.device_type
    
    if hasattr(tensor, "device_mesh"):
        return tensor.redistribute(mesh, placements)

    with torch.no_grad():
        if str(tensor.device).split(":")[0] != mesh_device_type:
            tensor = tensor.to(mesh_device_type)
        
        if not tensor.is_leaf:
            tensor = tensor.detach()

    return torch_distribute_tensor(tensor, mesh, placements)


def distribute_data_input(per_process_batch, layout, batch_dim_name):
    """Distribute the input data with the corresponding layout."""
    return distribute_tensor(per_process_batch, layout)


def parallelize_layer(layer, distribution):
    """Parallelize a layer using PyTorch parallelize_module."""
    from keras.src.distribution import ModelParallel
    from keras.src.backend.torch.core import Variable
    
    if not isinstance(distribution, ModelParallel):
        return

    if getattr(layer, "_is_parallelized", False):
        return

    mesh = distribution.device_mesh.backend_mesh
    layout_map = distribution._layout_map
    
    plan = {}
    variable_to_attr = {}
    
    def find_variables(obj, path_prefix=""):
        if isinstance(obj, Variable):
            return
            
        for name, child in obj.named_children():
            find_variables(child, path_prefix + name + ".")
            
        for name, param in obj.named_parameters(recurse=False):
            for var in layer.variables:
                if var.value is param:
                    style = _infer_parallel_style(var, layout_map, name)
                    if style:
                        variable_to_attr[var.path] = (var, obj, name, style)
                        break

    find_variables(layer)
    
    module_plans = {}
    for var_path, (var, module, attr_name, style) in variable_to_attr.items():
        if module not in module_plans:
            module_plans[module] = {}
        module_plans[module][attr_name] = style
        setattr(module, attr_name, var.value)

    for module, sub_plan in module_plans.items():
        if isinstance(module, torch.nn.ParameterDict):
            continue
        if DEBUG:
            print(f"[DEBUG] Parallelizing module {module} with plan {list(sub_plan.keys())}")
        parallelize_module(module, mesh, sub_plan)
        
    for var_path, (var, module, attr_name, style) in variable_to_attr.items():
        sharded_param = getattr(module, attr_name)
        if not hasattr(sharded_param, "placements"):
            layout = layout_map[var.path]
            sharded_param = distribute_variable(var.value, layout)
            setattr(module, attr_name, sharded_param)

        if not isinstance(sharded_param, Variable):
            var._value = sharded_param
            if not hasattr(sharded_param, "constraint"):
                sharded_param.constraint = var.constraint

    if hasattr(layer, "_torch_params"):
        for var in layer.variables:
            if var.path in layer.torch_params:
                layer.torch_params[var.path] = var.value

    layer._is_parallelized = True


def _infer_parallel_style(variable, layout_map, attr_name):
    """Infer PyTorch ParallelStyle from Keras LayoutMap."""
    layout = layout_map[variable.path]
    if layout is None or not any(axis is not None for axis in layout.axes):
        return None
        
    model_dim = "model"
    if model_dim in layout.axes:
        shard_idx = layout.axes.index(model_dim)
        if "kernel" in attr_name or "embeddings" in attr_name or "weight" in attr_name:
            if shard_idx == 1:
                return ColwiseParallel()
            elif shard_idx == 0:
                return RowwiseParallel()
        elif "bias" in attr_name:
            if shard_idx == 0:
                return ColwiseParallel()
    return None
