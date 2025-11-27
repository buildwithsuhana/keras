import keras
from keras import layers
from keras.src.distribution.tensor_parallel.tensor_layout import LayoutMap

def analyze_dense_layer(layer):
    if not isinstance(layer, layers.Dense): return 'dense'

    input_dim, output_dim = None, None
    kernel_var = None
    
    if hasattr(layer, "weights"):
        for w in layer.weights:
            if "kernel" in w.name:
                kernel_var = w
                break
    
    if kernel_var is None and hasattr(layer, 'kernel'): kernel_var = layer.kernel

    if kernel_var is not None and hasattr(kernel_var, 'shape'):
        input_dim, output_dim = kernel_var.shape[0], kernel_var.shape[1]

    if input_dim is None or output_dim is None:
        if hasattr(layer, 'units'): output_dim = layer.units
        if hasattr(layer, 'input_shape') and len(layer.input_shape) > 1: input_dim = layer.input_shape[-1]

    if not input_dim or not output_dim: return 'dense'

    if output_dim > input_dim * 1.25: return 'up_projection'
    elif input_dim > output_dim * 1.25: return 'down_projection'
    return 'dense'

def _get_variable_by_name_suffix(layer, suffix):
    if hasattr(layer, "weights"):
        for w in layer.weights:
            if w.name.endswith(suffix) or f"/{suffix}" in w.name: return w
    return None

def _apply_layer_sharding_rules(layer, layout_map):
    def safe_path(var): return getattr(var, "path", None)

    if isinstance(layer, layers.Dense):
        mlp_type = analyze_dense_layer(layer)
        kernel = _get_variable_by_name_suffix(layer, "kernel")
        bias = _get_variable_by_name_suffix(layer, "bias")
        k_path = safe_path(kernel) if kernel is not None else None
        b_path = safe_path(bias) if bias is not None else None

        if mlp_type == 'up_projection': 
            if k_path: layout_map[k_path] = (None, "model")
            if bias and b_path: layout_map[b_path] = ("model",)
        elif mlp_type == 'down_projection':
            if k_path: layout_map[k_path] = ("model", None)
            if bias and b_path: layout_map[b_path] = (None,)
            layout_map.output_rules[layer.name] = "allreduce sum"
        else:
            if k_path: layout_map[k_path] = (None, "model")
            if bias and b_path: layout_map[b_path] = ("model",)

    elif isinstance(layer, layers.EinsumDense):
        kernel = _get_variable_by_name_suffix(layer, "kernel")
        k_path = safe_path(kernel) if kernel is not None else None
        
        is_attn_output = "attention_output" in layer.name
        if not is_attn_output and k_path and "attention_output" in k_path: is_attn_output = True
            
        if is_attn_output:
            if k_path: layout_map[k_path] = ("model", None)
            layout_map.output_rules[layer.name] = "allreduce sum"
        else:
            if k_path: layout_map[k_path] = (None, "model")

    elif isinstance(layer, (layers.Embedding,)) or "Embedding" in layer.__class__.__name__:
        emb = _get_variable_by_name_suffix(layer, "embeddings")
        if emb is None: emb = _get_variable_by_name_suffix(layer, "weight")
        if emb is None and hasattr(layer, 'embeddings'): emb = layer.embeddings
            
        if emb is not None and safe_path(emb):
            layout_map[safe_path(emb)] = (None, "model")

def get_default_config(module, mesh):
    layout_map = LayoutMap(mesh)
    processed_layers = set()
    stack = [module]
    
    print("ℹ️ [AutoConfig] Scanning layer graph...")

    while stack:
        current_layer = stack.pop()
        if id(current_layer) in processed_layers: continue
        processed_layers.add(id(current_layer))

        _apply_layer_sharding_rules(current_layer, layout_map)

        children = []
        if hasattr(current_layer, 'layers') and current_layer.layers:
            children.extend(current_layer.layers)
        
        for attr in dir(current_layer):
            if attr.startswith('__'): continue
            try: val = getattr(current_layer, attr, None)
            except: continue
            if isinstance(val, layers.Layer): children.append(val)
            elif isinstance(val, (list, tuple)):
                for item in val:
                    if isinstance(item, layers.Layer): children.append(item)
        
        stack.extend(reversed(children))
        
    print(f"ℹ️ [AutoConfig] Rules generated for {len(layout_map)} tensors.")
    return layout_map