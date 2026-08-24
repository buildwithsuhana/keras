import functools

from keras.src import layers
from keras.src.backend import distribution_lib
from keras.src.distribution.tensor_parallel.tensor_layout import LayoutMap
from keras.src.distribution.tensor_parallel.tensor_layout import (
    split_tensor_for_parallelism,
)


def analyze_dense_layer(layer):
    """Classifies a Dense layer based on its input/output dimensions."""
    kernel = getattr(layer, "kernel", getattr(layer, "_kernel", None))
    if kernel is not None and len(kernel.shape) == 2:
        input_dim, output_dim = kernel.shape
        expansion_threshold = 1.5
        if output_dim > input_dim * expansion_threshold:
            return "up_projection"
        elif input_dim > output_dim * expansion_threshold:
            return "down_projection"
    return "dense"


def _reduce_sum(x):
    if isinstance(x, (list, tuple)):
        from keras.src import ops
        if len(x) == 1:
            return x[0]
        return ops.add_n(x)
    return distribution_lib.all_reduce(x, op="sum", axis_name="model")


def _gather(x, axis):
    if isinstance(x, (list, tuple)):
        from keras.src import ops
        if len(x) == 1:
            return x[0]
        return ops.concatenate(x, axis=axis)
    return distribution_lib.all_gather(x, axis=axis, axis_name="model")


def _apply_layer_sharding_rules(layer, device_count, state_rules, output_rules):
    """Applies sharding rules using object IDs for the requested log format."""

    def split_rule(dim):
        return functools.partial(
            split_tensor_for_parallelism, device_count=device_count, dim=dim
        )

    def gather_rule(axis):
        return functools.partial(_gather, axis=axis)

    layer_path = layer.path

    if isinstance(layer, layers.Dense):
        mlp_type = analyze_dense_layer(layer)
        if mlp_type == "up_projection":
            state_rules[id(layer.kernel)] = split_rule(dim=1)
            if layer.use_bias:
                state_rules[id(layer.bias)] = split_rule(dim=0)
        elif mlp_type == "down_projection":
            state_rules[id(layer.kernel)] = split_rule(dim=0)
            output_rules[layer_path] = _reduce_sum
        else:
            state_rules[id(layer.kernel)] = split_rule(dim=1)
            if layer.use_bias:
                state_rules[id(layer.bias)] = split_rule(dim=0)
            output_rules[layer_path] = gather_rule(axis=-1)

    elif isinstance(layer, layers.EinsumDense):
        if "attention_output" in layer.name:
            state_rules[id(layer.kernel)] = split_rule(dim=0)
            output_rules[layer_path] = _reduce_sum
        elif any(x in layer.name for x in ["query", "key", "value", "attention"]):
            state_rules[id(layer.kernel)] = split_rule(dim=1)
            if hasattr(layer, "bias") and layer.bias is not None:
                state_rules[id(layer.bias)] = split_rule(dim=0)
            output_rules[layer_path] = gather_rule(axis=-1)
        else:
            state_rules[id(layer.kernel)] = split_rule(dim=1)
            if hasattr(layer, "bias") and layer.bias is not None:
                state_rules[id(layer.bias)] = split_rule(dim=0)
            output_rules[layer_path] = gather_rule(axis=-1)

    elif (
        isinstance(layer, (layers.Embedding,))
        or "Embedding" in layer.__class__.__name__
        or hasattr(layer, "embeddings")
    ):
        emb = getattr(layer, "embeddings", None)
        if emb is not None:
            if "token_embedding" in layer.name:
                state_rules[id(emb)] = split_rule(dim=0)
                output_rules[layer_path] = _reduce_sum
            elif "position_embedding" in layer.name:
                state_rules[id(emb)] = split_rule(dim=1)
                output_rules[layer_path] = _reduce_sum
            else:
                state_rules[id(emb)] = split_rule(dim=0)
                output_rules[layer_path] = _reduce_sum

    elif isinstance(layer, layers.Dropout):
        output_rules[layer_path] = "parallel_dropout"


def get_default_config(model, device_ids):
    """Generates a default tensor parallelism configuration for a model."""
    device_count = len(device_ids)
    state_rules = {}
    output_rules = {}

    for layer in model._flatten_layers(recursive=True, include_self=True):
        _apply_layer_sharding_rules(
            layer, device_count, state_rules, output_rules
        )

    # Global model rules for final output
    output_rules[model.name] = _reduce_sum

    return LayoutMap(state_rules=state_rules, output_rules=output_rules)
