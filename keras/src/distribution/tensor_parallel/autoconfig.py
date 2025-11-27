import keras
from keras import layers
from keras.distribution import LayoutMap

def analyze_dense_layer(layer):
    """Classifies a Dense layer based on its input/output dimensions."""
    if not isinstance(layer, layers.Dense):
        return 'dense'

    input_dim = None
    output_dim = None

    # Try to infer shapes from built kernel or config
    if hasattr(layer, 'kernel') and layer.kernel is not None:
        input_dim, output_dim = layer.kernel.shape
    elif hasattr(layer, 'units') and hasattr(layer, 'input_shape') and layer.input_shape:
        output_dim = layer.units
        input_dim = layer.input_shape[-1]

    if not input_dim or not output_dim:
        return 'dense'

    expansion_threshold = 1.5
    if output_dim > input_dim * expansion_threshold:
        return 'up_projection'
    elif input_dim > output_dim * expansion_threshold:
        return 'down_projection'
    else:
        return 'dense'

def _apply_layer_sharding_rules(layer, full_name, device_count, layout_map):
    """Applies Keras Layout rules to a single layer."""
    
    # 1. Dense Layers (MLP / Projections)
    if isinstance(layer, layers.Dense):
        mlp_type = analyze_dense_layer(layer)

        if mlp_type == 'up_projection': # Column Parallel
            layout_map[f"{full_name}/kernel"] = (None, "model")
            if layer.use_bias:
                layout_map[f"{full_name}/bias"] = ("model",)
                
        elif mlp_type == 'down_projection': # Row Parallel
            layout_map[f"{full_name}/kernel"] = ("model", None)
            if layer.use_bias:
                # Bias is usually applied after AllReduce, so it's replicated
                layout_map[f"{full_name}/bias"] = (None,)
        else:
            # Default Dense (e.g. Heads): Column Parallel usually safer default
            layout_map[f"{full_name}/kernel"] = (None, "model")
            if layer.use_bias:
                layout_map[f"{full_name}/bias"] = ("model",)

    # 2. EinsumDense (Attention projections often use this in KerasNLP)
    elif isinstance(layer, layers.EinsumDense):
        # Heuristic: Output projections often end with 'attention_output'
        if "attention_output" in full_name: # Row Parallel
            layout_map[f"{full_name}/kernel"] = ("model", None)
            if hasattr(layer, 'bias') and layer.bias is not None:
                layout_map[f"{full_name}/bias"] = (None,)
        else: # Query/Key/Value projections: Column Parallel
            layout_map[f"{full_name}/kernel"] = (None, "model")
            if hasattr(layer, 'bias') and layer.bias is not None:
                layout_map[f"{full_name}/bias"] = ("model",)

    # 3. Embeddings
    elif isinstance(layer, (layers.Embedding,)) or "Embedding" in layer.__class__.__name__:
        # Vocab splitting (Column Parallel equivalent)
        layout_map[f"{full_name}/embeddings"] = (None, "model")
        layout_map[f"{full_name}/weights"] = (None, "model")


def get_default_config(module, mesh):
    """Generates a Keras LayoutMap for the model."""
    layout_map = LayoutMap(mesh)
    
    # [FIX] Correctly access mesh shape via index
    if "model" in mesh.axis_names:
        model_idx = mesh.axis_names.index("model")
        device_count = mesh.shape[model_idx]
    else:
        device_count = 1
    
    processed_layers = set()
    stack = [(module, "")]

    while stack:
        current_layer, prefix = stack.pop()
        if id(current_layer) in processed_layers: continue
        processed_layers.add(id(current_layer))

        # Keras variable paths usually use '/' instead of '.'
        name = current_layer.name
        full_name = f"{prefix}/{name}" if prefix else name

        _apply_layer_sharding_rules(current_layer, full_name, device_count, layout_map)

        # Recurse children
        children = []
        
        # Standard sub-layers
        if hasattr(current_layer, 'layers'):
            children.extend([(l, full_name) for l in current_layer.layers])
            
        # Specific attributes for Backbones (e.g. GemmaBackbone)
        for attr in ['token_embedding', 'embeddings']:
            if hasattr(current_layer, attr):
                val = getattr(current_layer, attr)
                if isinstance(val, layers.Layer):
                    children.append((val, full_name))

        stack.extend(reversed(children))

    return layout_map