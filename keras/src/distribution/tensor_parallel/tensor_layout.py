"""
This file defines the layout map and the sharding rule object.
"""

import collections
from keras.src import ops


def split_tensor_for_parallelism(tensor, index, device_count, dim):
    """Calculates a slice of a tensor along a specified dimension for a
    given index.

    This utility is used in tensor parallelism API to distribute a
    tensor across multiple devices.

    Args:
        tensor: The full tensor to be sharded.
        index: The index of the device/shard to return (e.g., 0, 1, 2...).
        device_count: The total number of parallel devices or splits.
        dim: The dimension along which to split the tensor. If -1, the
            last dimension is used.

    Returns:
        A tensor slice corresponding to the given `index`.
    """
    if dim == -1:
        split_dim = ops.ndim(tensor) - 1
    else:
        split_dim = dim

    splits = ops.array_split(
        tensor, indices_or_sections=device_count, axis=split_dim
    )
    return splits[index]


# --- MODIFIED CLASS ---
# This class replaces the simple lambda rule. It is still callable,
# but now it also stores the dimension, which is critical for
# saving and loading checkpoints.
class SplitRule:
    """A callable rule that splits a tensor and stores the split dimension."""
    def __init__(self, dim, device_count):
        self.dim = dim
        self.device_count = device_count
        # --- REMOVED self.sharding_type ---

    def __call__(self, tensor, index):
        """Executes the split function."""
        return split_tensor_for_parallelism(
            tensor, index, self.device_count, self.dim
        )

    def __repr__(self):
        return f"<SplitRule(dim={self.dim}, devices={self.device_count})>"

# --- END MODIFIED CLASS ---


LayoutMap = collections.namedtuple("LayoutMap", ["state_rules", "output_rules"])