import re
import numpy as np
from keras.src import layers
from keras.src.distribution.tensor_parallel.tensor_layout import LayoutMap

def _split_tensor_cpu(x, index, device_count, dim):
    """Splits the tensor using standard NumPy on CPU to avoid GPU OOM."""
    splits = np.array_split(x, device_count, axis=dim)
    return splits[index]

def _split_rule(device_count, dim):
    return lambda x, index: _split_tensor_cpu(x, index, device_count, dim=dim)

def _apply_layer_sharding_rules(layer, full_name, device_count, state_rules, output_rules):
    lname = layer.name.lower() if layer.name else ""
    cls_name = layer.__class__.__name__
    
    # DEBUG: Use forward slashes to match Keras Hub Variable paths (e.g. 'layer/kernel')
    # This prevents the 'NoneType' errors caused by variables remaining on CPU.
    clean_name = full_name.replace(".", "/").lstrip("/")
    
    print(f"🔍 [AutoConfig] Analyzing Layer: '{clean_name}' ({cls_name})")

    # 1. Shard Normalization (Gamma/Beta/Scale)
    # Gemma often uses 'scale' for RMSNorm and 'gamma'/'beta' for LayerNorm.
    if any(x in cls_name for x in ["Normalization", "LayerNorm", "RMSNorm"]):
        for attr in ["scale", "gamma", "beta"]:
            key = f"{clean_name}/{attr}"
            state_rules[key] = _split_rule(device_count, dim=0)
            print(f"   [DEBUG] Normalization Rule Created: {key} (Split Dim 0)")
        return

    # 2. Shard Dense / EinsumDense Layers
    if "Dense" in cls_name:
        is_down_proj = any(x in lname for x in ["down_proj", "output", "o_proj", "ffw_linear"])
        is_up_proj = any(x in lname for x in ["up_proj", "gate", "ffw_gating"])
        is_qkv = any(x in lname for x in ["query", "key", "value", "q_proj", "k_proj", "v_proj"])
        
        is_3d_kernel = False
        if hasattr(layer, 'kernel') and layer.kernel is not None:
            if len(layer.kernel.shape) == 3:
                is_3d_kernel = True

        if is_down_proj:
            # Kernel: (Input, Output) -> Split Output (Dim 1) to keep shards independent
            split_dim = len(layer.kernel.shape) - 1 if is_3d_kernel else 1
            key = f"{clean_name}/kernel"
            state_rules[key] = _split_rule(device_count, dim=split_dim)
            print(f"   [DEBUG] Dense (DOWN) Rule Created: {key} (Split Dim {split_dim})")
            
        elif is_up_proj or is_qkv:
            # Kernel: (Input, Output) -> Split Input (Dim 0)
            split_dim = 1 if is_3d_kernel else 0
            key = f"{clean_name}/kernel"
            state_rules[key] = _split_rule(device_count, dim=split_dim)
            print(f"   [DEBUG] Dense (UP/QKV) Rule Created: {key} (Split Dim {split_dim})")
            
        else:
            # Fallback for generic Dense
            key = f"{clean_name}/kernel"
            state_rules[key] = _split_rule(device_count, dim=0)
            print(f"   [DEBUG] Dense (FALLBACK) Rule Created: {key} (Split Dim 0)")

    # 3. Embedding Rules
    elif "Embedding" in cls_name:
        # Hub models use 'embeddings' or 'weight'
        for attr in ["embeddings", "weight"]:
            key = f"{clean_name}/{attr}"
            state_rules[key] = _split_rule(device_count, dim=1)
            print(f"   [DEBUG] Embedding Rule Created: {key} (Split Dim 1)")
        
        # Disable Gather to keep the output sharded for the next layer
        output_rules[clean_name] = {0: "no_comm"}
        print(f"   [DEBUG] Output Rule Created: {clean_name} -> no_comm")

def get_default_config(module, device_ids):
    print(f"\n🚀 [AutoConfig] Starting generation for model: {module.name}")
    device_count = len(device_ids)
    state_rules = {}
    output_rules = {}
    processed_layers = set()
    stack = [(module, None)]

    while stack:
        current_layer, prefix = stack.pop()
        if id(current_layer) in processed_layers: continue
        processed_layers.add(id(current_layer))

        name = current_layer.name
        if prefix is None:
            full_name = ""
        elif "backbone" in current_layer.__class__.__name__.lower():
            full_name = prefix
        else:
            full_name = f"{prefix}.{name}" if prefix else name
        
        # Deduplicate names in the path to handle recursive lookups
        parts = full_name.split('.')
        clean_parts = []
        for p in parts:
            if not clean_parts or clean_parts[-1] != p: clean_parts.append(p)
        full_name = ".".join(clean_parts)

        _apply_layer_sharding_rules(current_layer, full_name, device_count, state_rules, output_rules)

        # Traverse layer attributes to find child layers
        children_to_add = []
        for attr_name in dir(current_layer):
            if attr_name.startswith('__') or attr_name.startswith('_'): continue
            if attr_name in ['trainable_variables', 'non_trainable_variables', 'weights', 'variables']: continue
            try: 
                attr_value = getattr(current_layer, attr_name, None)
            except: 
                continue
            
            if attr_value is None: continue

            # Check if attribute is a Keras Layer
            if hasattr(attr_value, "name") and isinstance(attr_value, layers.Layer):
                if attr_value is not current_layer:
                    children_to_add.append((attr_value, full_name))
            elif isinstance(attr_value, (list, tuple)):
                for item in attr_value:
                    if hasattr(item, "name") and isinstance(item, layers.Layer):
                        children_to_add.append((item, full_name))
        
        # Explicit check for .layers property
        if hasattr(current_layer, 'layers') and current_layer.layers:
            for sub_layer in current_layer.layers:
                if not any(sub_layer is existing[0] for existing in children_to_add):
                    children_to_add.append((sub_layer, full_name))
        
        stack.extend(reversed(children_to_add))

    print(f"✅ [AutoConfig] Generated {len(state_rules)} sharding rules.\n")
    return LayoutMap(state_rules=state_rules, output_rules=output_rules)