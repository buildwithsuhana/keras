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

def analyze_dense_layer(layer):
    if "Dense" not in layer.__class__.__name__:
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
    if not input_dim or not output_dim: return 'dense'
    expansion_threshold = 1.5
    if output_dim > input_dim * expansion_threshold: return 'up_projection'
    elif input_dim > output_dim * expansion_threshold: return 'down_projection'
    return 'dense'

def _apply_layer_sharding_rules(layer, full_name, device_count, state_rules, output_rules):
    lname = layer.name.lower() if layer.name else ""
    cls_name = layer.__class__.__name__
    clean_name = full_name.lstrip(".")
    rule_key_kernel = f"{clean_name}.kernel"
    rule_key_bias = f"{clean_name}.bias"
    
    # Normalization Rules keys
    rule_key_scale = f"{clean_name}.scale"
    rule_key_gamma = f"{clean_name}.gamma"
    rule_key_beta = f"{clean_name}.beta"

    print(f"🔍 [AutoConfig] Analyzing: '{clean_name}' ({cls_name})")

    # 1. Shard Normalization (Crucial for input shape consistency)
    if "Normalization" in cls_name:
        print(f"   ➕ [Add Rule] '{clean_name}' -> Normalization (Split Dim 0)")
        def add_if_exists(attr, key):
            if hasattr(layer, attr) and getattr(layer, attr) is not None:
                state_rules[key] = _split_rule(device_count, dim=0)
        add_if_exists("scale", rule_key_scale)
        add_if_exists("gamma", rule_key_gamma)
        add_if_exists("beta", rule_key_beta)
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
            # Down Projection (Output Processing)
            # Input is Full Heads/Features -> We want Sharded Output (1792)
            
            if is_3d_kernel:
                # Kernel: (Heads, HeadDim, Output) or (Heads, Output, HeadDim)?
                # Usually EinsumDense output is last dim.
                split_dim = len(layer.kernel.shape) - 1 
            else:
                # Kernel: (Input, Output) -> Split Output (Dim 1)
                split_dim = 1

            print(f"   ➕ [Add Rule] '{clean_name}' -> Down Projection (Split Output Dim {split_dim})")
            state_rules[rule_key_kernel] = _split_rule(device_count, dim=split_dim)
            
            # Disable AllReduce to keep shards independent
            # output_rules[clean_name] = {0: "allreduce"} 
            
        elif is_up_proj or is_qkv:
            # Up Projection / QKV (Input Processing)
            # Input is Sharded (1792) -> We must split the Input Dimension of the kernel.
            
            if is_3d_kernel:
                # Kernel: (Heads, Input, HeadDim) -> Input is Dim 1
                split_dim = 1
            else:
                # Kernel: (Input, Output) -> Input is Dim 0
                split_dim = 0
            
            print(f"   ➕ [Add Rule] '{clean_name}' -> Up Projection/QKV (Split Input Dim {split_dim})")
            state_rules[rule_key_kernel] = _split_rule(device_count, dim=split_dim)
            
            # Bias usually matches Output/Heads. Since we split Input, 
            # Output is Full. So Bias should NOT be sharded (Replicated).
            if hasattr(layer, "bias") and layer.bias is not None:
                pass # Default is replicated, which is correct here.
            
        else:
            # Fallback for generic Dense (treat as Up/Expansion)
            print(f"   ➕ [Add Rule] '{clean_name}' -> Dense Fallback (Split Input)")
            split_dim = 0
            state_rules[rule_key_kernel] = _split_rule(device_count, dim=split_dim)

    # 3. Embedding
    elif "Embedding" in cls_name:
        if hasattr(layer, 'weights'):
            any_sharded = False
            for weight in layer.weights:
                if "embedding" in weight.name or "weight" in weight.name:
                    attr_name = None
                    for candidate in ['embeddings', 'token_embedding', 'position_embedding', 'weight']:
                        if getattr(layer, candidate, None) is weight:
                            attr_name = candidate
                            break
                    if not attr_name:
                        attr_name = weight.name.split('/')[-1].split(':')[0]

                    # Decide whether to replicate the full embedding on each device (safe for token
                    # embeddings) or to column-split it (column parallel) for memory savings.
                    is_token = 'token' in clean_name.lower() or 'token' in (lname or '') or 'token' in attr_name.lower()
                    if is_token:
                        print(f"   ➕ [Add Rule] '{clean_name}' -> Embedding ({attr_name}) (Replicated)")
                        state_rules[f"{clean_name}.{attr_name}"] = (lambda x, index: x)
                    else:
                        print(f"   ➕ [Add Rule] '{clean_name}' -> Embedding ({attr_name}) (Column Parallel)")
                        state_rules[f"{clean_name}.{attr_name}"] = _split_rule(device_count, dim=1)
                        any_sharded = True

            # If at least one weight in this embedding layer is sharded, keep the layer marked
            # as "no_comm" (no gather). If all embeddings were replicated, we don't force no_comm.
            if any_sharded:
                print(f"   ➕ [Output Rule] '{clean_name}' -> No Comm (Keep Sharded)")
                output_rules[clean_name] = {0: "no_comm"}

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
        
        parts = full_name.split('.')
        clean_parts = []
        for p in parts:
            if not clean_parts or clean_parts[-1] != p: clean_parts.append(p)
        full_name = ".".join(clean_parts)

        _apply_layer_sharding_rules(current_layer, full_name, device_count, state_rules, output_rules)

        children_to_add = []
        for attr_name in dir(current_layer):
            if attr_name.startswith('__') or attr_name.startswith('_'): continue
            if attr_name in ['trainable_variables', 'non_trainable_variables', 'weights', 'variables']: continue
            try: attr_value = getattr(current_layer, attr_name, None)
            except: continue
            if attr_value is None: continue

            if hasattr(attr_value, "name") and "Layer" in attr_value.__class__.__bases__[0].__name__:
                if attr_value is not current_layer:
                    children_to_add.append((attr_value, full_name))
            elif isinstance(attr_value, (list, tuple)):
                for item in attr_value:
                    if hasattr(item, "name") and "Layer" in item.__class__.__bases__[0].__name__:
                        children_to_add.append((item, full_name))
        
        if hasattr(current_layer, 'layers') and current_layer.layers:
            for sub_layer in current_layer.layers:
                is_duplicate = False
                for existing_child, _ in children_to_add:
                    if existing_child is sub_layer:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    children_to_add.append((sub_layer, full_name))
        stack.extend(reversed(children_to_add))

    print(f"✅ [AutoConfig] Generated {len(state_rules)} sharding rules.\n")
    return LayoutMap(state_rules=state_rules, output_rules=output_rules)