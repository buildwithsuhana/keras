import re
from keras.src import layers
from keras.src.distribution.tensor_parallel.tensor_layout import (
    split_tensor_for_parallelism,
    LayoutMap
)

_split_fn_internal = split_tensor_for_parallelism


def _split_rule(device_count, dim):
    """Creates a sharding rule for a specific dimension."""
    return lambda x, index: _split_fn_internal(x, index, device_count, dim=dim)


def analyze_dense_layer(layer):
    """
    Classifies a Dense layer based on its input/output dimensions.
    """
    if not isinstance(layer, layers.Dense):
        return 'dense'

    input_dim = None
    output_dim = None

    if hasattr(layer, 'kernel') and layer.kernel is not None:
        kernel_shape = layer.kernel.shape
        if len(kernel_shape) == 2:
            input_dim = kernel_shape[0]
            output_dim = kernel_shape[1]

    if input_dim is None or output_dim is None:
        if hasattr(layer, 'units'):
            output_dim = layer.units
        else:
            return 'dense'

        if hasattr(layer, 'input_shape') and layer.input_shape and len(layer.input_shape) > 1:
            input_dim = layer.input_shape[-1]
        else:
            return 'dense'

    if not input_dim or not output_dim:
        return 'dense'

    expansion_threshold = 1.5
    is_expansion = output_dim > input_dim * expansion_threshold
    is_contraction = input_dim > output_dim * expansion_threshold

    if is_expansion:
        return 'up_projection'
    elif is_contraction:
        return 'down_projection'
    else:
        return 'dense'


def _apply_layer_sharding_rules(layer, full_name, device_count, state_rules, output_rules):
    """
    Helper function that applies rules to a single layer instance.
    """
    # Helper to check names case-insensitively
    lname = layer.name.lower() if layer.name else ""
    
    # --- 1. DENSE / EINSUM DENSE LAYERS ---
    if isinstance(layer, (layers.Dense, layers.EinsumDense)):
        # Identify Layer Type by Name (Gemma / Llama / Standard Transformers)
        is_down_proj = any(x in lname for x in ["down_proj", "output", "o_proj", "ffw_linear"])
        is_up_proj = any(x in lname for x in ["up_proj", "gate", "ffw_gating"])
        is_qkv = any(x in lname for x in ["query", "key", "value", "q_proj", "k_proj", "v_proj"])
        
        # --- Strategy A: Down Projections (Row Parallel) ---
        if is_down_proj:
            state_rules[f"{full_name}.kernel"] = _split_rule(device_count, dim=0)
            output_rules[f"{full_name}"] = {0: "allreduce"}
            
        # --- Strategy B: Up Projections & QKV (Column Parallel) ---
        elif is_up_proj or is_qkv:
            state_rules[f"{full_name}.kernel"] = _split_rule(device_count, dim=1)
            if hasattr(layer, "bias") and layer.bias is not None:
                state_rules[f"{full_name}.bias"] = _split_rule(device_count, dim=0)
            output_rules[f"{full_name}"] = {0: "gather -1"}
            
        # --- Strategy C: Fallback (Heuristic based on Shape) ---
        elif isinstance(layer, layers.Dense):
            mlp_type = analyze_dense_layer(layer)
            if mlp_type == 'up_projection':
                state_rules[f"{full_name}.kernel"] = _split_rule(device_count, dim=1)
                if layer.use_bias:
                    state_rules[f"{full_name}.bias"] = _split_rule(device_count, dim=0)
                output_rules[f"{full_name}"] = {0: "gather"}
            elif mlp_type == 'down_projection':
                state_rules[f"{full_name}.kernel"] = _split_rule(device_count, dim=0)
                output_rules[f"{full_name}"] = {0: "allreduce"}
            else:
                state_rules[f"{full_name}.kernel"] = _split_rule(device_count, dim=1)
                if layer.use_bias:
                    state_rules[f"{full_name}.bias"] = _split_rule(device_count, dim=0)
                output_rules[f"{full_name}"] = {0: "gather -1"}
        
        # --- Strategy D: Fallback for EinsumDense (Default to Column) ---
        else:
            state_rules[f"{full_name}.kernel"] = _split_rule(device_count, dim=1)
            if hasattr(layer, 'bias') and layer.bias is not None:
                state_rules[f"{full_name}.bias"] = _split_rule(device_count, dim=0)
            output_rules[f"{full_name}"] = {0: "gather -1"}

    # --- 2. EMBEDDING LAYERS ---
    elif isinstance(layer, (layers.Embedding,)) or "Embedding" in layer.__class__.__name__:
        if hasattr(layer, 'weights'):
            for weight in layer.weights:
                if "embedding" in weight.name or "weight" in weight.name:
                    attr_name = None
                    for candidate in ['embeddings', 'token_embedding', 'position_embedding', 'weight']:
                        if getattr(layer, candidate, None) is weight:
                            attr_name = candidate
                            break
                    
                    if not attr_name:
                        attr_name = weight.name.split('/')[-1].split(':')[0]

                    state_rules[f"{full_name}.{attr_name}"] = _split_rule(device_count, dim=1)

            output_rules[f"{full_name}"] = {0: "gather -1"}


def get_default_config(module, device_ids):
    """
    Generates a default tensor parallelism configuration for a model.
    Fixes path generation to match Keras variable names robustly.
    """
    device_count = len(device_ids)
    state_rules = {}
    output_rules = {}
    
    processed_layers = set()
    
    # Start with None as prefix to indicate the root layer
    stack = [(module, None)]

    while stack:
        current_layer, prefix = stack.pop()

        if id(current_layer) in processed_layers:
            continue
        processed_layers.add(id(current_layer))

        name = current_layer.name
        
        # LOGIC CHANGE: 
        # 1. If prefix is None (Root), start with empty string.
        # 2. If layer is a Backbone, DO NOT append its name. 
        #    This allows the regex to match 'decoder_block_0' directly, which works
        #    whether the variable path is 'backbone/decoder_block_0' or just 'decoder_block_0'.
        
        if prefix is None:
            full_name = ""
        elif "Backbone" in current_layer.__class__.__name__:
            full_name = prefix  # Skip appending backbone name
        else:
            full_name = f"{prefix}.{name}" if prefix else name
        
        # Clean up repeated parts (e.g. gemma.gemma) just in case
        parts = full_name.split('.')
        clean_parts = []
        for p in parts:
            if not clean_parts or clean_parts[-1] != p:
                clean_parts.append(p)
        full_name = ".".join(clean_parts)

        # Apply Rules
        _apply_layer_sharding_rules(
            current_layer, full_name, device_count, state_rules, output_rules
        )

        children_to_add = []

        # 1. Standard Layers traversal
        if hasattr(current_layer, 'layers') and current_layer.layers:
            for sub_layer in current_layer.layers:
                children_to_add.append((sub_layer, full_name))

        # 2. Special Attributes traversal
        for specific_attr in ['token_embedding', 'embeddings', 'position_embedding', 'backbone', 'transformer_layers']:
            if hasattr(current_layer, specific_attr):
                attr_val = getattr(current_layer, specific_attr)
                if isinstance(attr_val, layers.Layer):
                    children_to_add.append((attr_val, full_name))
                elif isinstance(attr_val, (list, tuple)):
                    for i, item in enumerate(attr_val):
                        if isinstance(item, layers.Layer):
                            children_to_add.append((item, f"{full_name}"))

        stack.extend(reversed(children_to_add))

    return LayoutMap(
        state_rules=state_rules,
        output_rules=output_rules
    )