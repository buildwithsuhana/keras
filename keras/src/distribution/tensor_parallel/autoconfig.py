from keras.src import layers
from keras.src.distribution.tensor_parallel.tensor_layout import (
    split_tensor_for_parallelism,
    LayoutMap
)

# Alias for internal split function
_split_fn_internal = split_tensor_for_parallelism


def _split_rule(device_count, dim):
    """
    Creates a sharding rule for a specific dimension.
    """
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
    if isinstance(layer, layers.Dense):
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

    elif isinstance(layer, layers.EinsumDense):
        if "attention_output" in full_name:
            state_rules[f"{full_name}.kernel"] = _split_rule(device_count, dim=0)
            output_rules[f"{full_name}"] = {0: "allreduce"}
        else:
            state_rules[f"{full_name}.kernel"] = _split_rule(device_count, dim=1)
            if hasattr(layer, 'bias') and layer.bias is not None:
                state_rules[f"{full_name}.bias"] = _split_rule(device_count, dim=0)
            output_rules[f"{full_name}"] = {0: "gather -1"}

    elif isinstance(layer, (layers.Embedding,)):
        weight_name = None 
        if hasattr(layer, 'embeddings'):
            weight_name = 'embeddings'
        elif hasattr(layer, 'position_embeddings'):
            weight_name = 'position_embeddings'

        if weight_name:
            state_rules[f"{full_name}.{weight_name}"] = _split_rule(device_count, dim=1)
            output_rules[f"{full_name}"] = {0: "no_comm"}


def get_default_config_keras(module, device_ids):
    """
    Generates a default tensor parallelism configuration for a model using
    iterative graph traversal (stack-based).
    """
    device_count = len(device_ids)
    state_rules = {}
    output_rules = {}
    
    processed_layers = set()
    
    stack = [(module, "")]

    while stack:
        current_layer, prefix = stack.pop()

        if id(current_layer) in processed_layers:
            continue
        processed_layers.add(id(current_layer))

        name = current_layer.name
        full_name = f"{prefix}.{name}" if prefix else name

        _apply_layer_sharding_rules(
            current_layer, full_name, device_count, state_rules, output_rules
        )

        children_to_add = []

        # 1. Check standard Keras sub-layers list
        if hasattr(current_layer, 'layers') and current_layer.layers:
            for sub_layer in current_layer.layers:
                children_to_add.append((sub_layer, full_name))

        # 2. [FIX] Explicitly check for common NLP attributes (e.g. OPT/GPT backbones)
        # This ensures 'token_embedding' is found and sharded, preventing replication.
        for specific_attr in ['token_embedding', 'embeddings', 'position_embedding']:
            if hasattr(current_layer, specific_attr):
                attr_val = getattr(current_layer, specific_attr)
                if isinstance(attr_val, layers.Layer):
                    children_to_add.append((attr_val, full_name))

        # 3. Check attributes for hidden layers (common in Subclassing API)
        for attr_name in dir(current_layer):
            if attr_name.startswith('__') and attr_name.endswith('__'):
                continue
            
            # Optimization: Skip known non-layer attributes
            if attr_name in ['trainable_variables', 'non_trainable_variables', 'weights']:
                continue

            try:
                if not hasattr(current_layer, attr_name):
                    continue
                attr = getattr(current_layer, attr_name)
            except Exception:
                continue

            if isinstance(attr, layers.Layer) and attr is not current_layer:
                children_to_add.append((attr, full_name))
            elif isinstance(attr, (list, tuple)):
                for item in attr:
                    if isinstance(item, layers.Layer):
                        children_to_add.append((item, full_name))
        
        stack.extend(children_to_add)

    return LayoutMap(
        state_rules=state_rules,
        output_rules=output_rules
    )