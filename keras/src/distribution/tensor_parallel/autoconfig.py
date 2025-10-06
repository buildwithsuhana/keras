from typing import Sequence, Dict, Any, Set

from keras.src.distribution.tensor_parallel.config import ConfigKeras
from keras.src.distribution.tensor_parallel.state_action_keras import SplitKeras


def analyze_dense_layer_directly(layer, module, prefix: str) -> str:
    """Analyzes a Dense layer to classify it for tensor parallelism sharding.

    This function inspects the layer's weight shapes to determine if it's an
    "up-projection" (expanding feature dimensions), a "down-projection"
    (contracting feature dimensions), or a generic layer. This classification
    helps in deciding whether to apply column-wise or row-wise parallelism.

    Args:
        layer: The keras.layers.Dense instance to analyze.
        module: The parent Keras model containing the layer.
        prefix: The hierarchical name prefix for the layer.

    Returns:
        A string indicating the layer's classification: 'up_projection',
        'down_projection', or 'generic_dense'.
    """
    from keras.src import layers

    if not isinstance(layer, layers.Dense):
        return 'generic_dense'

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
            return 'generic_dense'

        if hasattr(layer, 'input_shape') and layer.input_shape and len(layer.input_shape) > 1:
            input_dim = layer.input_shape[-1]
        else:
            return 'generic_dense'

    if not input_dim or not output_dim:
        return 'generic_dense'

    expansion_threshold = 1.5
    is_expansion = output_dim > input_dim * expansion_threshold
    is_contraction = input_dim > output_dim * expansion_threshold

    if is_expansion:
        return 'up_projection'
    elif is_contraction:
        return 'down_projection'
    else:
        return 'generic_dense'


def _find_and_shard_layers(
    current_layer: "layers.Layer",
    prefix: str,
    module: "layers.Layer",
    world_size: int,
    state_rules: Dict[str, Any],
    output_rules: Dict[str, Any],
    processed_layers: Set[int],
):
    """Recursively traverses a Keras model to generate sharding rules.

    This is an internal helper function that navigates through all layers of a
    model, including nested ones. For each supported layer, it determines the
    appropriate sharding strategy and populates the `state_rules` and
    `output_rules` dictionaries. These dictionaries are modified in place.

    Args:
        current_layer: The Keras layer to be processed in the current step.
        prefix: The hierarchical name prefix for the `current_layer`, used to
            build the full unique name of a layer (e.g., "encoder.block_1").
        module: The top-level Keras model being analyzed. This is passed down
            for context.
        world_size: The total number of devices to shard the model across.
        state_rules: A dictionary that is populated with sharding rules for the
            model's weights (state). The keys are regex patterns matching
            weight names, and values are sharding actions.
        output_rules: A dictionary that is populated with communication rules
            for the outputs of layers (e.g., all-gather, all-reduce).
        processed_layers: A set containing the IDs of layers that have already
            been processed. This prevents infinite loops in models with shared
            layers.
    """
    from keras.src import layers

    if id(current_layer) in processed_layers:
        return
    processed_layers.add(id(current_layer))

    name = current_layer.name
    full_name = f"{prefix}.{name}" if prefix else name

    if isinstance(current_layer, layers.Dense):
        mlp_type = analyze_dense_layer_directly(current_layer, module, full_name)
        
        if mlp_type == 'up_projection':
            state_rules[f"^{full_name}.kernel$"] = SplitKeras(world_size, 1, "column")
            if current_layer.use_bias:
                state_rules[f"^{full_name}.bias$"] = SplitKeras(world_size, 0, "column")
            output_rules[f"^{full_name}$"] = {0: "gather"}
        
        elif mlp_type == 'down_projection':
            state_rules[f"^{full_name}.kernel$"] = SplitKeras(world_size, 0, "row")
            output_rules[f"^{full_name}$"] = {0: "allreduce"}
        
        else:
            state_rules[f"^{full_name}.kernel$"] = SplitKeras(world_size, 1, "column")
            if current_layer.use_bias:
                state_rules[f"^{full_name}.bias$"] = SplitKeras(world_size, 0, "column") 
            output_rules[f"^{full_name}$"] = {0: "gather -1"}
        return

    elif isinstance(current_layer, layers.EinsumDense):
        if "attention_output" in full_name:
            state_rules[f"^{full_name}.kernel$"] = SplitKeras(world_size, 0, "row")
            if hasattr(current_layer, 'bias') and current_layer.bias is not None:
                pass
            output_rules[f"^{full_name}$"] = {0: "allreduce"}
        else:
            state_rules[f"^{full_name}.kernel$"] = SplitKeras(world_size, 1, "column")
            if hasattr(current_layer, 'bias') and current_layer.bias is not None:
                state_rules[f"^{full_name}.bias$"] = SplitKeras(world_size, 0, "column")
            output_rules[f"^{full_name}$"] = {0: "gather -1"}
        return 

    elif isinstance(current_layer, (layers.Embedding,)):
        if hasattr(current_layer, 'token_embedding') or hasattr(current_layer, 'position_embedding'):
            pass
        else:
            weight_name = None
            if hasattr(current_layer, 'embeddings'):
                weight_name = 'embeddings'
            elif hasattr(current_layer, 'position_embeddings'):
                weight_name = 'position_embeddings'
            
            if weight_name:
                state_rules[f"^{full_name}\..*{weight_name}$"] = SplitKeras(world_size, 1, "column")
                output_rules[f"^{full_name}$"] = {0: "no_comm"} 
            return

    elif isinstance(current_layer, (layers.LayerNormalization, layers.BatchNormalization, layers.GroupNormalization)):
        return
    
    if hasattr(current_layer, 'layers') and current_layer.layers:
        for sub_layer in current_layer.layers:
            _find_and_shard_layers(
                sub_layer, full_name, module, world_size, 
                state_rules, output_rules, processed_layers
            )
    
    for attr_name in dir(current_layer):
        if attr_name.startswith('__') and attr_name.endswith('__'):
            continue
        if hasattr(current_layer, attr_name):
            attr = getattr(current_layer, attr_name)

            if isinstance(attr, layers.Layer) and attr is not current_layer:
                _find_and_shard_layers(
                    attr, full_name, module, world_size, 
                    state_rules, output_rules, processed_layers
                )
            elif isinstance(attr, (list, tuple)):
                for item in attr:
                    if isinstance(item, layers.Layer):
                        _find_and_shard_layers(
                            item, full_name, module, world_size,
                            state_rules, output_rules, processed_layers
                        )

def get_default_config_keras(module, device_ids: Sequence[str]) -> ConfigKeras:
    """Generates a default sharding configuration for a Keras model.

    This function serves as the main entry point for automatically creating a
    tensor parallelism sharding configuration. It initializes the data
    structures for holding the sharding rules and then calls a recursive
    helper function (`_find_and_shard_layers`) to traverse the model and
    populate these rules.

    Args:
        module: The Keras model or layer to be configured for sharding.
        device_ids: A sequence of device ID strings (e.g., `['gpu:0', 'gpu:1']`)
            to shard across. The number of devices determines the `world_size`.

    Returns:
        A `ConfigKeras` object containing the generated `state_rules` for
        sharding weights and `output_rules` for handling communications
        for layer outputs.
    """
    world_size = len(device_ids)
    state_rules = {}
    output_rules = {}
    processed_layers = set()

    _find_and_shard_layers(
        current_layer=module, 
        prefix="", 
        module=module,
        world_size=world_size,
        state_rules=state_rules,
        output_rules=output_rules,
        processed_layers=processed_layers
    ) 
    
    return ConfigKeras(
        state_rules=state_rules,
        output_rules=output_rules
    )