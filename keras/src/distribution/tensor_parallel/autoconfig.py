import keras
from keras import layers
# Import LOCAL tensor_layout to get the dict-based LayoutMap
from keras.src.distribution.tensor_parallel.tensor_layout import LayoutMap

def analyze_dense_layer(layer):
    """Classifies a Dense layer based on its input/output dimensions."""
    if not isinstance(layer, layers.Dense): return 'dense'
    
    # ... (Keep existing size detection logic) ...
    input_dim, output_dim = None, None
    if hasattr(layer, 'kernel') and hasattr(layer.kernel, 'shape'):
         input_dim, output_dim = layer.kernel.shape
    
    if not input_dim or not output_dim: return 'dense'

    if output_dim > input_dim * 1.25: return 'up_projection'
    elif input_dim > output_dim * 1.25: return 'down_projection'
    return 'dense'

def _apply_layer_sharding_rules(layer, layout_map):
    """Applies sharding AND communication rules."""
    
    def safe_path(var): return getattr(var, "path", None)
    
    # 1. Dense Layers
    if isinstance(layer, layers.Dense):
        mlp_type = analyze_dense_layer(layer)
        
        # Helper to find kernel/bias
        kernel = next((w for w in layer.weights if "kernel" in w.name), None)
        bias = next((w for w in layer.weights if "bias" in w.name), None)
        
        k_path = safe_path(kernel)
        b_path = safe_path(bias)

        if mlp_type == 'up_projection': 
            # Column Parallel (Split Output/Dim 1)
            if k_path: layout_map[k_path] = (None, "model")
            if b_path: layout_map[b_path] = ("model",) # Bias matches output shard

        elif mlp_type == 'down_projection':
            # Row Parallel (Split Input/Dim 0) -> Output needs Reduction
            if k_path: layout_map[k_path] = ("model", None)
            if b_path: layout_map[b_path] = (None,) # Bias is added after reduction usually
            
            # Add Reduction Rule
            layout_map.output_rules[layer.name] = "allreduce sum"

        else:
            # Default to Column Parallel for general dense
            if k_path: layout_map[k_path] = (None, "model")

    # 2. Embedding
    elif isinstance(layer, layers.Embedding) or "Embedding" in layer.__class__.__name__:
        # Find embedding weight
        w = next((x for x in layer.weights if "embedding" in x.name or "weight" in x.name), None)
        if w and safe_path(w):
            layout_map[safe_path(w)] = (None, "model")

    # 3. EinsumDense (Attention)
    elif isinstance(layer, layers.EinsumDense):
        # ... (Keep existing logic) ...
        kernel = next((w for w in layer.weights if "kernel" in w.name), None)
        k_path = safe_path(kernel)
        
        if k_path:
             if "attention_output" in k_path or "attention_output" in layer.name:
                 # Row Parallel -> Reduce
                 layout_map[k_path] = ("model", None)
                 layout_map.output_rules[layer.name] = "allreduce sum"
             else:
                 # Column Parallel (QKV)
                 layout_map[k_path] = (None, "model")


def get_default_config(module, mesh):
    layout_map = LayoutMap(mesh)
    
    # BFS/DFS to traverse layers
    stack = [module]
    visited = set()
    
    while stack:
        layer = stack.pop()
        if id(layer) in visited: continue
        visited.add(id(layer))
        
        _apply_layer_sharding_rules(layer, layout_map)
        
        # Add children
        if hasattr(layer, 'layers'):
            stack.extend(layer.layers)
            
        # Check attributes for nested layers (common in NLP models)
        for attr in dir(layer):
            val = getattr(layer, attr, None)
            if isinstance(val, layers.Layer):
                stack.append(val)
                
    return layout_map