import jax
import jax.lax as lax
import jax.numpy as jnp
import keras

def get_device_info():
    """Retrieves information about the available JAX devices."""
    available_devices = jax.devices()
    return {
        "backend": "jax",
        "devices": [str(d) for d in available_devices],
        "device_count": len(available_devices),
    }

def is_multi_device_capable():
    """Checks if more than one JAX device is available."""
    return jax.local_device_count() > 1

def get_communication_ops():
    """
    Provides a dictionary of JAX collective communication operations.
    These functions are designed to be called from within a `pmap` or `pjit` context.
    """

    def all_reduce(x, op="sum", axis_name="data"):
        """Reduces a tensor across all devices along a pmap axis."""
        reduce_ops = {
            "sum": lax.psum,
            "mean": lax.pmean,
        }
        reduce_fn = reduce_ops.get(op)
        if reduce_fn is None:
            raise ValueError(f"Unsupported all_reduce op: {op}")
        return reduce_fn(x, axis_name=axis_name)

    def all_gather(x, axis=0, axis_name="data"):
        """Gathers tensors from all devices along a pmap axis."""
        return lax.all_gather(x, axis_name=axis_name, axis=axis)

    def broadcast(x, root=0, axis_name="data"):
        """Broadcasts a tensor from a root device to all other devices."""
        return lax.all_gather(x, axis_name=axis_name, axis=0)[root]

    def scatter(x, root=0, axis=0, axis_name="data"):
        """Scatters a tensor from a root device to all devices."""
        full_tensor = lax.all_gather(x, axis_name=axis_name, axis=0)[root]

        device_id = lax.axis_index(axis_name=axis_name)
        num_devices = lax.psum(1, axis_name=axis_name)
        chunk_size = full_tensor.shape[axis] // num_devices
        start_index = device_id * chunk_size

        return lax.dynamic_slice_in_dim(
            operand=full_tensor,
            start_index=start_index,
            slice_size=chunk_size,
            axis=axis,
        )

    return {
        "all_reduce": all_reduce,
        "all_gather": all_gather,
        "broadcast": broadcast,
        "scatter": scatter,
    }