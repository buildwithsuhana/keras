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
    # Heuristic to detect projection type based on shape expansion
    if "Dense" not in layer.__class__.__name__: return 'dense'
    if not hasattr(layer, 'kernel') or layer.kernel is None: return 'dense'
    
    shape = layer.kernel.shape
    if len(shape) != 2: return 'dense'
    
    input_dim, output_dim = shape[0], shape[1]
    if output_dim > input_dim * 1.5: return 'up_projection'   # e.g. FFW Up
    if input_dim > output_dim * 1.5: return 'down_projection' # e.g. FFW Down
    return 'dense'

def _apply_layer_sharding_rules(layer, full_name, device_count, state_rules, output_rules):
    lname = layer.name.lower() if layer.name else ""
    cls_name = layer.__class__.__name__
    clean_name = full_name.lstrip(".")
    rule_key_kernel = f"{clean_name}.kernel"
    rule_key_bias = f"{clean_name}.bias"

    print(f"🔍 [AutoConfig] Analyzing: '{clean_name}' ({cls_name})")

    if "Dense" in cls_name:
        is_down_proj = any(x in lname for x in ["down_proj", "output", "o_proj", "ffw_linear"])
        is_up_proj = any(x in lname for x in ["up_proj", "gate", "ffw_gating"])
        is_qkv = any(x in lname for x in ["query", "key", "value", "q_proj", "k_proj", "v_proj"])
        
        # Check for 3D kernel (EinsumDense with Heads)
        is_3d_kernel = False
        if hasattr(layer, 'kernel') and layer.kernel is not None:
            if len(layer.kernel.shape) == 3:
                is_3d_kernel = True

        if is_down_proj:
            # STRATEGY: Down Projection (Restoring Sharded State)
            # Input: Full (from previous Op). Output: H/N (Sharded).
            # Split Kernel on Output Dimension.
            split_dim = 2 if is_3d_kernel else 1
            print(f"   ➕ [Rule] '{clean_name}' -> Down Projection (Split Output Dim {split_dim})")
            
            state_rules[rule_key_kernel] = _split_rule(device_count, dim=split_dim)
            if hasattr(layer, "bias") and layer.bias is not None:
                # Bias is (Output_Dim,). Since output is sharded, bias must be sharded on dim 0.
                state_rules[rule_key_bias] = _split_rule(device_count, dim=0)
            
            # Output Rule: None. We want the output to remain sharded.
            
        elif is_up_proj or is_qkv:
            # STRATEGY: Up Projection (Consuming Sharded State)
            # Input: H/N (Sharded). Output: Full (Intermediate).
            # Split Kernel on Input Dimension (Contraction Dim).
            split_dim = 0 # Input is always dim 0 for Dense/EinsumDense contraction here
            
            print(f"   ➕ [Rule] '{clean_name}' -> Up Projection/QKV (Split Input Dim {split_dim}) + AllReduce")
            
            state_rules[rule_key_kernel] = _split_rule(device_count, dim=split_dim)
            # Bias is (Output_Dim,). Output is Full. Bias must be Replicated (dim=None or don't shard).
            # No bias rule needed -> defaults to replicated.

            # Output Rule: Since we split the contraction dimension, we have partial sums.
            # We must AllReduce to get the correct values.
            output_rules[clean_name] = {0: "allreduce"}
            
        elif "EinsumDense" not in cls_name:
            # Fallback for standard Dense layers not matching names
            mlp_type = analyze_dense_layer(layer)
            if mlp_type == 'up_projection':
                # Treat as Up Projection (Split Input, AllReduce Output)
                print(f"   ➕ [Rule] '{clean_name}' -> Dense (Up-Proj Heuristic)")
                state_rules[rule_key_kernel] = _split_rule(device_count, dim=0)
                output_rules[clean_name] = {0: "allreduce"}
            elif mlp_type == 'down_projection':
                # Treat as Down Projection (Split Output, No Comm)
                print(f"   ➕ [Rule] '{clean_name}' -> Dense (Down-Proj Heuristic)")
                state_rules[rule_key_kernel] = _split_rule(device_count, dim=1)
                if layer.use_bias:
                    state_rules[rule_key_bias] = _split_rule(device_count, dim=0)
            else:
                # Square matrix? Default to Row Parallel style (Split Output)?
                # Let's default to preserving Sharded State (Split Output).
                state_rules[rule_key_kernel] = _split_rule(device_count, dim=1)
                if layer.use_bias:
                    state_rules[rule_key_bias] = _split_rule(device_count, dim=0)
        else:
            # Fallback for generic EinsumDense
            print(f"   ➕ [Rule] '{clean_name}' -> EinsumDense Fallback (Split Input)")
            # Assume it's consuming sharded input
            state_rules[rule_key_kernel] = _split_rule(device_count, dim=0)
            output_rules[clean_name] = {0: "allreduce"}

    elif "Embedding" in cls_name:
        if hasattr(layer, 'weights'):
            for weight in layer.weights:
                if "embedding" in weight.name or "weight" in weight.name:
                    attr_name = weight.name.split('/')[-1].split(':')[0]
                    if not hasattr(layer, attr_name) and hasattr(layer, f"_{attr_name}"):
                        attr_name = f"_{attr_name}"
                    
                    print(f"   ➕ [Rule] Embedding '{clean_name}' -> Split Hidden Dim 1")
                    state_rules[f"{clean_name}.{attr_name}"] = _split_rule(device_count, dim=1)
                    # Output is H/N. No rule needed.

    elif "Normalization" in cls_name:
        found_norm_weight = False
        for attr in ["scale", "gamma", "beta"]:
            if hasattr(layer, attr) and getattr(layer, attr) is not None:
                print(f"   ➕ [Rule] Norm '{clean_name}' -> Split {attr} Dim 0")
                state_rules[f"{clean_name}.{attr}"] = _split_rule(device_count, dim=0)
                found_norm_weight = True
        
        if not found_norm_weight:
             print(f"   ⚠️ [Warning] Normalization layer '{clean_name}' found but no known weights detected.")

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