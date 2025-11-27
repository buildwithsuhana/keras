import keras
from keras import layers
from keras.distribution import LayoutMap

def analyze_dense_layer(layer):
    """
    Classifies a Dense layer based on its input/output dimensions.
    """
    if not isinstance(layer, layers.Dense):
        return 'dense'

    input_dim = None
    output_dim = None

    # Try to infer shapes safely
    kernel_var = None
    if hasattr(layer, "weights"):
        for w in layer.weights:
            if "kernel" in w.name:
                kernel_var = w
                break
    
    # Fallback to property
    if kernel_var is None and hasattr(layer, 'kernel'):
        kernel_var = layer.kernel

    if kernel_var is not None and hasattr(kernel_var, 'shape'):
        kernel_shape = kernel_var.shape
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
    if output_dim > input_dim * expansion_threshold:
        return 'up_projection'
    elif input_dim > output_dim * expansion_threshold:
        return 'down_projection'
    else:
        return 'dense'

def _get_variable_by_name_suffix(layer, suffix):
    """Helper to find a variable in a layer by its name suffix (e.g. 'kernel')."""
    if hasattr(layer, "weights"):
        for w in layer.weights:
            # Check both exact suffix and path-style suffix
            if w.name.endswith(suffix) or f"/{suffix}" in w.name:
                return w
    return None

def _apply_layer_sharding_rules(layer, layout_map):
    """
    Applies Keras LayoutMap rules to a single layer instance.
    """
    
    def safe_path(var):
        return getattr(var, "path", None)

    # 1. Dense Layers
    if isinstance(layer, layers.Dense):
        mlp_type = analyze_dense_layer(layer)
        
        # Find variables safely
        kernel_var = _get_variable_by_name_suffix(layer, "kernel")
        bias_var = _get_variable_by_name_suffix(layer, "bias")

        kernel_path = safe_path(kernel_var) if kernel_var is not None else None
        bias_path = safe_path(bias_var) if bias_var is not None else None

        if mlp_type == 'up_projection': 
            # Column Parallel
            if kernel_path: layout_map[kernel_path] = (None, "model")
            if layer.use_bias and bias_path: layout_map[bias_path] = ("model",)

        elif mlp_type == 'down_projection':
            # Row Parallel
            if kernel_path: layout_map[kernel_path] = ("model", None)
            if layer.use_bias and bias_path: layout_map[bias_path] = (None,)

        else:
            # Default Dense
            if kernel_path: layout_map[kernel_path] = (None, "model")
            if layer.use_bias and bias_path: layout_map[bias_path] = ("model",)

    # 2. EinsumDense
    elif isinstance(layer, layers.EinsumDense):
        kernel_var = _get_variable_by_name_suffix(layer, "kernel")
        bias_var = _get_variable_by_name_suffix(layer, "bias")
        
        kernel_path = safe_path(kernel_var) if kernel_var is not None else None
        bias_path = safe_path(bias_var) if bias_var is not None else None

        # Robust check for attention output
        is_attn_output = "attention_output" in layer.name
        if not is_attn_output and kernel_path and "attention_output" in kernel_path:
            is_attn_output = True
            
        if is_attn_output:
            if kernel_path: layout_map[kernel_path] = ("model", None)
            if bias_path: layout_map[bias_path] = (None,)
        else:
            if kernel_path: layout_map[kernel_path] = (None, "model")
            if bias_path: layout_map[bias_path] = ("model",)

    # 3. Embeddings
    elif isinstance(layer, (layers.Embedding,)) or "Embedding" in layer.__class__.__name__:
        emb_var = _get_variable_by_name_suffix(layer, "embeddings")
        if emb_var is None: 
            emb_var = _get_variable_by_name_suffix(layer, "weight")
        
        # Fallback
        if emb_var is None and hasattr(layer, 'embeddings'):
            if hasattr(layer.embeddings, 'path'):
                emb_var = layer.embeddings
            
        # [FIX] Explicit check 'is not None' to avoid TypeError on Variable boolean check
        if emb_var is not None and safe_path(emb_var):
            layout_map[safe_path(emb_var)] = (None, "model")


def get_default_config(module, mesh):
    """
    Generates a Keras LayoutMap using robust graph traversal.
    """
    layout_map = LayoutMap(mesh)
    
    processed_layers = set()
    stack = [module]

    while stack:
        current_layer = stack.pop()

        if id(current_layer) in processed_layers:
            continue
        processed_layers.add(id(current_layer))

        # Apply rules to the current layer
        _apply_layer_sharding_rules(current_layer, layout_map)

        # Add children layers to stack
        children_to_add = []

        # Standard sub-layers
        if hasattr(current_layer, 'layers') and current_layer.layers:
            children_to_add.extend(current_layer.layers)

        # Specific attributes for Backbones
        for specific_attr in ['token_embedding', 'embeddings', 'position_embedding']:
            if hasattr(current_layer, specific_attr):
                attr_val = getattr(current_layer, specific_attr)
                if isinstance(attr_val, layers.Layer):
                    children_to_add.append(attr_val)

        # Scan for other attributes
        for attr_name in dir(current_layer):
            if attr_name.startswith('__') or attr_name in ['trainable_variables', 'weights', 'layers', 'variables']:
                continue
            try:
                attr_value = getattr(current_layer, attr_name, None)
            except:
                continue

            if isinstance(attr_value, layers.Layer) and attr_value is not current_layer:
                children_to_add.append(attr_value)
            elif isinstance(attr_value, (list, tuple)):
                for item in attr_value:
                    if isinstance(item, layers.Layer):
                        children_to_add.append(item)
        
        stack.extend(reversed(children_to_add))

    return layout_map
    