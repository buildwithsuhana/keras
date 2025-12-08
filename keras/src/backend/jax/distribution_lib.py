"""Utilities for distribution strategy with JAX backend.

This file contains the core JAX distribution primitives from Keras,
along with higher-level device management and auto-configuration utilities.
This version does not use try-except blocks for error handling.
"""

import logging
from typing import Dict
from typing import List
from typing import Optional

import jax
import jax.lax as lax
import numpy as np

from keras.src.backend.common import global_state
from keras.src.random import seed_generator
from keras.src.utils import jax_utils
from keras.src.utils import rng_utils

logger = logging.getLogger(__name__)


def list_devices(device_type=None):
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
    jax_devices = jax.devices(backend=device_type)
    return [f"{device.platform}:{device.id}" for device in jax_devices]


def get_device_count():
    """Returns the number of local JAX devices."""
    return jax.local_device_count()


def get_device_info(device_id: str) -> Dict[str, any]:
    """Get detailed information about a specific device."""
    device_info = {
        "id": device_id,
        "type": None,
        "index": None,
        "memory": None,
        "capabilities": None,
    }

    device_type, device_index = device_id.split(":")
    device_info["type"] = device_type.upper()
    device_info["index"] = int(device_index)

    return device_info


def get_best_devices(count: int = 1) -> List[str]:
    """Get the best available devices for tensor parallelism."""
    all_devices = list_devices()

    if count <= 0:
        return []

    if count > len(all_devices):
        logger.warning(
            f"Requested {count} devices but only {len(all_devices)} available"
        )
        count = len(all_devices)

    return all_devices[:count]


def get_device_backend(device_type: str) -> str:
    """Get the recommended backend for a device type."""
    backend_mapping = {"tpu": "jax", "gpu": "jax", "cpu": "jax"}
    return backend_mapping.get(device_type.lower(), "jax")


def validate_device_placement(device_id: str) -> bool:
    """Validate if a device can be used for tensor operations."""
    all_devices = list_devices()
    return device_id in all_devices


def get_device_memory_info(device_id: str) -> Optional[Dict[str, any]]:
    """Get memory information for a device (if available)."""
    if device_id.startswith("gpu:"):
        return {
            "type": "GPU",
            "index": int(device_id.split(":")[1]),
            "memory": "Available",
        }
    elif device_id.startswith("tpu:"):
        return {
            "type": "TPU",
            "index": int(device_id.split(":")[1]),
            "memory": "TPU Memory",
        }
    elif device_id.startswith("cpu:"):
        return {
            "type": "CPU",
            "index": int(device_id.split(":")[1]),
            "memory": "System RAM",
        }

    return None


def auto_configure_tensor_parallel(
    world_size: int = None, backend: str = None
) -> Dict[str, any]:
    """Automatically configure tensor parallelism with the best available devices."""
    all_devices = list_devices()

    if not all_devices:
        raise RuntimeError("No devices available for tensor parallelism")

    if world_size is None:
        world_size = len(all_devices)
    else:
        world_size = min(world_size, len(all_devices))

    selected_devices = all_devices[:world_size]

    recommended_backend = "jax"

    config = {
        "devices": selected_devices,
        "world_size": world_size,
        "backend": recommended_backend,
    }

    logger.info(f"Auto-configured tensor parallelism: {config}")
    return config


def distribute_variable(value, layout):
    """Create a distributed variable for JAX."""
    return distribute_tensor(value, layout)


def distribute_tensor(tensor, layout):
    """Distribute the tensor based on the layout."""
    # Avoid circular imports.
    from keras.src.distribution import TensorLayout

    if isinstance(layout, TensorLayout):
        layout = layout.backend_layout

    if jax_utils.is_in_jax_tracing_scope():
        return jax.lax.with_sharding_constraint(tensor, layout)

    # Skip relayout if unnecessary.
    if isinstance(tensor, jax.Array):
        if isinstance(
            layout, jax.sharding.Sharding
        ) and tensor.sharding.is_equivalent_to(layout, ndim=len(tensor.shape)):
            return tensor
        elif hasattr(layout, "layout"):
            current_layout = getattr(tensor, "layout", None)
            if current_layout == layout:
                return tensor
        elif hasattr(layout, "format"):
            current_layout = getattr(tensor, "format", None)
            if current_layout == layout:
                return tensor

    return jax.device_put(tensor, layout)


def distribute_data_input(per_process_batch, layout, batch_dim_name):
    """Distribute the input data with the corresponding layout."""
    # Avoid circular imports.
    from keras.src.distribution import TensorLayout

    if isinstance(layout, TensorLayout):
        layout = layout.backend_layout

    return jax.make_array_from_process_local_data(layout, per_process_batch)


def initialize_rng():
    """Initializes the global random number generator across processes."""
    global_seed = rng_utils.get_random_seed()
    if global_seed is None:
        cpu_devices = jax.devices("cpu")
        num_local_cpu_devices = jax.local_device_count("cpu")
        local_seed = jax.numpy.asarray(
            [seed_generator.make_default_seed()] * num_local_cpu_devices,
            dtype=jax.numpy.uint32,
        )
        global_seed = jax.pmap(
            lambda x: jax.lax.psum(x, "all"),
            axis_name="all",
            devices=cpu_devices,
        )(local_seed).item(0)
        rng_utils.set_random_seed(global_seed)

    global_seed_generator = global_state.get_global_attribute(
        "global_seed_generator"
    )
    if global_seed_generator is not None:
        seed = global_seed_generator.get_config()["seed"]
        if seed is None:
            global_state.set_global_attribute(
                "global_seed_generator",
                seed_generator.SeedGenerator(
                    seed=global_seed,
                    name=global_seed_generator.name,
                    backend=global_seed_generator.backend,
                ),
            )


def initialize(job_addresses, num_processes, process_id):
    if job_addresses and "," in job_addresses:
        job_addresses = job_addresses.split(",")
        if num_processes is not None and num_processes != len(job_addresses):
            raise ValueError(
                f"The provided job_addresses {job_addresses} has "
                f"{len(job_addresses)} jobs, but num_processes is "
                f"{num_processes}"
            )
        coordinator_address = job_addresses[0]
    else:
        coordinator_address = job_addresses

    jax.distributed.initialize(
        coordinator_address=coordinator_address,
        num_processes=num_processes,
        process_id=process_id,
    )
    initialize_rng()


def num_processes():
    return jax.process_count()


def process_id():
    return jax.process_index()


def all_reduce(x, op="sum", axis_name="model"):
    if op == "sum":
        return lax.psum(x, axis_name=axis_name)
    elif op == "mean":
        sum_val = lax.psum(x, axis_name=axis_name)
        axis_size = lax.psum(1, axis_name=axis_name) 
        return sum_val / axis_size
    else:
        raise ValueError(
            f"Unsupported reduction operation: {op}. "
            "Supported options are 'sum' and 'mean'."
        )


def all_gather(x, axis, axis_name="model"):
    return lax.all_gather(x, axis_name=axis_name, axis=axis, tiled=True)


def _to_backend_device(device_name):
    if isinstance(device_name, jax.Device):
        return device_name
    device_name = str(device_name)
    if ":" not in device_name:
        device_type, device_id = device_name, 0
    else:
        device_type, device_id = device_name.split(":")

    # FIX: Allow searching for 'gpu' if 'cuda' is requested
    search_backend = 'gpu' if device_type == 'cuda' else device_type
    
    try:
        devices = jax.devices(backend=search_backend)
    except RuntimeError:
        devices = jax.devices()

    for device in devices:
        # FIX: Relaxed platform check to match cuda <-> gpu
        platform_match = (device.platform == device_type) or \
                         (device.platform == 'gpu' and device_type == 'cuda')
        
        if platform_match and device.id == int(device_id):
            return device
            
    raise ValueError(f"Device not found: {device_name}")


def _to_backend_mesh(device_mesh):
    shape = device_mesh.devices.shape
    devices = [_to_backend_device(d) for d in device_mesh.devices.flatten()]
    devices = np.array(devices).reshape(shape)
    return jax.sharding.Mesh(devices, device_mesh.axis_names)


def _to_backend_layout(tensor_layout):
    if tensor_layout.device_mesh is None:
        raise ValueError(
            "Cannot create sharding when device mesh is not set "
            "for TensorLayout."
        )
    partition_spec = jax.sharding.PartitionSpec(*tensor_layout.axes)
    jax_mesh = tensor_layout.device_mesh.backend_mesh
    return jax.sharding.NamedSharding(jax_mesh, partition_spec)