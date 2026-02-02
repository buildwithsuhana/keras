"""Path conversion utilities for distribution strategies.

This module provides utilities for converting between Keras and PyTorch
variable path naming conventions.
"""

import re


def keras_to_pytorch_path(keras_path: str) -> str:
    """Convert a Keras variable path to PyTorch format.

    Args:
        keras_path: Path like "dense/kernel" or "my_model/dense_1/bias"

    Returns:
        PyTorch-style path like "dense.weight" or "my_model.dense_1.bias"

    Example:
        >>> keras_to_pytorch_path("dense/kernel")
        'dense.weight'
        >>> keras_to_pytorch_path("conv2d/bias")
        'conv2d.bias'
    """
    # Replace forward slashes with dots
    pytorch_path = keras_path.replace('/', '.')
    return pytorch_path


def pytorch_to_keras_path(pytorch_path: str) -> str:
    """Convert a PyTorch variable path to Keras format.

    Args:
        pytorch_path: Path like "dense.weight" or "my_model.dense_1.bias"

    Returns:
        Keras-style path like "dense/kernel" or "my_model/dense_1/bias"

    Example:
        >>> pytorch_to_keras_path("dense.weight")
        'dense/kernel'
        >>> pytorch_to_keras_path("conv2d.bias")
        'conv2d/bias'
    """
    # Replace dots with forward slashes
    keras_path = pytorch_path.replace('.', '/')
    return keras_path


def convert_path_for_matching(
    path: str,
    source_format: str = "keras",
) -> tuple:
    """Convert a path to work with regex patterns from the other format.

    This is useful when you have regex patterns in Keras format but need
    to match against PyTorch paths (and vice versa).

    Args:
        path: The path to convert
        source_format: "keras" if path is in Keras format, "pytorch" if in PyTorch

    Returns:
        Tuple of (keras_path, pytorch_path)
    """
    if source_format == "keras":
        keras_path = path
        pytorch_path = keras_to_pytorch_path(path)
    else:
        pytorch_path = path
        keras_path = pytorch_to_keras_path(path)

    return keras_path, pytorch_path
