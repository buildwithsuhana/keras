"""Utilities for distribution strategy with Torch backend.

This file contains the core Torch distribution primitives for explicit 
parameter sharding, bypassing DTensor auto-propagation for performance.
"""

import logging
import os
from typing import List

import numpy as np
import torch
import torch.distributed as dist

from keras.src.backend.common import global_state
from keras.src.random import seed_generator
from keras.src.utils import rng_utils

logger = logging.getLogger(__name__)


def list_devices(device_type=None):
    """Return all available global devices for distributed computation."""
    if device_type:
        device_type = device_type.lower()
    else:
        device_type = "cuda" if torch.cuda.is_available() else "cpu"

    if device_type in ("gpu", "cuda"):
        if not torch.cuda.is_available():
            return []
        return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    elif device_type == "cpu":
        return ["cpu:0"]
    return []


def get_best_devices(count: int = 1) -> List[str]:
    """Get the best available devices for explicit tensor parallelism."""
    all_devices = list_devices("cuda")
    if not all_devices:
        all_devices = list_devices("cpu")

    if count <= 0:
        return []
    if count > len(all_devices):
        count = len(all_devices)

    return all_devices[:count]


def distribute_variable(value, layout):
    """Bypasses DTensor if sharding is handled explicitly."""
    return value


def distribute_tensor(tensor, layout):
    """Bypasses DTensor for explicit tensor operations."""
    return tensor


def distribute_data_input(per_process_batch, layout, batch_dim_name):
    """Bypasses DTensor for data input."""
    return per_process_batch


def initialize_rng():
    """Initializes the global RNG across processes to ensure consistency."""
    global_seed = rng_utils.get_random_seed()
    if global_seed is None:
        if not dist.is_initialized():
            seed = seed_generator.make_default_seed()
        else:
            if process_id() == 0:
                seed = seed_generator.make_default_seed()
                seed_tensor = torch.tensor(seed, dtype=torch.int64, device="cpu")
            else:
                seed_tensor = torch.empty(1, dtype=torch.int64, device="cpu")
            dist.broadcast(seed_tensor, src=0)
            seed = seed_tensor.item()
        global_seed = seed
        rng_utils.set_random_seed(global_seed)

    global_seed_generator = global_state.get_global_attribute(
        "global_seed_generator"
    )
    if global_seed_generator is not None and global_seed_generator.seed is None:
        global_state.set_global_attribute(
            "global_seed_generator",
            seed_generator.SeedGenerator(
                seed=global_seed,
                name=global_seed_generator.name,
                backend=global_seed_generator.backend,
            ),
        )


def initialize(job_addresses, num_processes, process_id):
    """Initializes the distributed process group in PyTorch."""
    os.environ["RANK"] = str(process_id)
    os.environ["WORLD_SIZE"] = str(num_processes)

    if job_addresses:
        master_addr = job_addresses.split(",")[0]
        if ":" in master_addr:
            host, port = master_addr.split(":")
            os.environ["MASTER_ADDR"] = host
            os.environ["MASTER_PORT"] = port

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    initialize_rng()


def num_processes():
    """Return the number of processes for the current setting."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def process_id():
    """Return the current process ID for the setting."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def all_reduce(x, op="sum", axis_name="model"):
    """Reduces a tensor using standard PyTorch NCCL/Gloo collectives."""
    if not dist.is_initialized():
        return x

    reduce_op = dist.ReduceOp.SUM if op == "sum" else dist.ReduceOp.AVG
    x_out = x.clone()
    dist.all_reduce(x_out, op=reduce_op)
    return x_out


def all_gather(x, axis, axis_name="model"):
    """Gathers tensors from all devices using standard PyTorch collectives."""
    if not dist.is_initialized():
        return x

    world_size = dist.get_world_size()
    gather_list = [torch.empty_like(x) for _ in range(world_size)]
    dist.all_gather(gather_list, x)
    return torch.cat(gather_list, dim=axis)


def _to_backend_mesh(device_mesh):
    """Maps Keras DeviceMesh to a flattened representation for PyTorch."""
    return device_mesh.devices.flatten().tolist()


def _to_backend_layout(tensor_layout):
    """No-op for explicit sharding."""
    return None