import functools
from keras.src import layers
from keras.src.backend import distribution_lib
from keras.src.distribution.tensor_parallel.tensor_layout import LayoutMap
from keras.src.distribution.tensor_parallel.tensor_layout import (
    split_tensor_for_parallelism,
)

def analyze_dense_layer(layer):
    input_dim, output_dim = None, None
    kernel = getattr(layer, "kernel", getattr(layer, "_kernel", None))
    if kernel is not None and len(kernel.shape) == 2:
        input_dim, output_dim = kernel.shape[0], kernel.shape[1]
    if output_dim is None and hasattr(layer, "units"):
        output_dim = layer.units
    if input_dim is None and hasattr(layer, "input_shape") and layer.input_shape:
        input_dim = layer.input_shape[-1]
    if input_dim is None or output_dim is None: return "dense"
    threshold = 1.5
    if output_dim > input_dim * threshold: return "up_projection"
    if input_dim > output_dim * threshold: return "down_projection"
    return "dense"

def _reduce_sum(x):
    return distribution_lib.all_reduce(x, op="sum", axis_name="model")

def _gather(x, axis):
    return distribution_lib.all_gather(x, axis=axis, axis_name="model")

def _apply_layer_sharding_rules(layer, device_count, state_rules, output_rules):
    def split_rule(dim):
        return functools.partial(split_tensor_for_parallelism, device_count=device_count, dim=dim)
    def gather_rule(axis):
        return functools.partial(_gather, axis=axis)

    layer_path = layer.path
    # 1. Core Layers
    if isinstance(layer, layers.Dense):
        mlp_type = analyze_dense_layer(layer)
        if mlp_type in ["up_projection", "dense"]:
            state_rules[layer.kernel.path] = split_rule(dim=1)
            if layer.use_bias: state_rules[layer.bias.path] = split_rule(dim=0)
            output_rules[layer_path] = gather_rule(axis=-1)
        elif mlp_type == "down_projection":
            state_rules[layer.kernel.path] = split_rule(dim=0)
            output_rules[layer_path] = _reduce_sum
    elif isinstance(layer, layers.EinsumDense):
        if "attention_output" in layer.name:
            state_rules[layer.kernel.path] = split_rule(dim=0)
            output_rules[layer_path] = _reduce_sum
        else:
            state_rules[layer.kernel.path] = split_rule(dim=1)
            if hasattr(layer, "bias") and layer.bias is not None:
                state_rules[layer.bias.path] = split_rule(dim=0)
            output_rules[layer_path] = gather_rule(axis=-1)
    elif isinstance(layer, (layers.Embedding,)) or "Embedding" in layer.__class__.__name__:
        emb = getattr(layer, "embeddings", None)
        if emb is not None: state_rules[emb.path] = split_rule(dim=1)
        output_rules[layer_path] = lambda x: x
    
    # 2. Normalization Layers (CRITICAL FIX)
    elif "Normalization" in layer.__class__.__name__:
        for attr in ["scale", "gamma", "beta", "bias"]:
            var = getattr(layer, attr, None)
            if var is not None:
                # Normalization weights are 1D, split along dim 0
                state_rules[var.path] = split_rule(dim=0)
        output_rules[layer_path] = lambda x: x

def get_default_config(model, device_ids):
    device_count = len(device_ids)
    state_rules, output_rules = {}, {}
    for layer in model._flatten_layers(recursive=True, include_self=True):
        _apply_layer_sharding_rules(layer, device_count, state_rules, output_rules)
    return LayoutMap(state_rules=state_rules, output_rules=output_rules)