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
    clean_name = full_name.lstrip(".")
    rule_key_kernel = f"{clean_name}.kernel"

    # 1. Shard Normalization (Crucial for sharded tensor support)
    if "Normalization" in cls_name:
        for attr in ["scale", "gamma", "beta"]:
            if hasattr(layer, attr) and getattr(layer, attr) is not None:
                state_rules[f"{clean_name}.{attr}"] = _split_rule(device_count, dim=0)
        return

    # 2. Shard Dense / EinsumDense Layers
    if "Dense" in cls_name:
        is_proj = any(x in lname for x in ["proj", "output", "gate", "ffw", "query", "key", "value"])
        if is_proj:
            # Persistent Row-Parallel: Always split the input dimension
            state_rules[rule_key_kernel] = _split_rule(device_count, dim=0)

    elif "Embedding" in cls_name:
        for v in layer.variables:
            attr = v.path.split("/")[-1] if hasattr(v, "path") else v.name.split("/")[-1].split(":")[0]
            state_rules[f"{clean_name}.{attr}"] = _split_rule(device_count, dim=1)
        output_rules[clean_name] = {0: "no_comm"}

def get_default_config(module, device_ids):
    device_count = len(device_ids)
    state_rules, output_rules, processed_layers = {}, {}, set()
    stack = [(module, "")]
    while stack:
        current_layer, prefix = stack.pop()
        if id(current_layer) in processed_layers: continue
        processed_layers.add(id(current_layer))
        name = current_layer.name
        full_name = f"{prefix}.{name}" if prefix else name
        full_name = ".".join([p for i, p in enumerate(full_name.split('.')) if i == 0 or p != full_name.split('.')[i-1]])
        _apply_layer_sharding_rules(current_layer, full_name, device_count, state_rules, output_rules)
        for attr_name in dir(current_layer):
            if attr_name.startswith('_'): continue
            try:
                attr_value = getattr(current_layer, attr_name)
                if isinstance(attr_value, layers.Layer) and attr_value is not current_layer:
                    stack.append((attr_value, full_name))
                elif isinstance(attr_value, (list, tuple)):
                    for item in attr_value:
                        if isinstance(item, layers.Layer): stack.append((item, full_name))
            except: continue
    return LayoutMap(state_rules=state_rules, output_rules=output_rules)