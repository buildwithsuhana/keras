import logging
import re
import gc
from typing import Any, Tuple, Set, Callable, TYPE_CHECKING
from keras import device, ops
import keras
import numpy as np

if TYPE_CHECKING:
    from keras.src.models import Model

logger = logging.getLogger(__name__)

class ParameterShardingStrategy:
    def __init__(self, device_count: int, rank: int):
        self.device_count = device_count
        self.rank = rank

    def _map_variables_to_owners(self, model):
        """
        Creates a mapping from variable object ID to (layer, attribute_name).
        This allows us to replace the variable on the layer if we need to resize it.
        """
        var_to_owner = {}
        
        # Traverse all layers
        stack = [model]
        visited = set()
        
        while stack:
            layer = stack.pop()
            if id(layer) in visited:
                continue
            visited.add(id(layer))

            # Inspect all attributes of the layer to find variables
            # We look at __dict__ to find the actual name the user/keras uses
            for attr_name, attr_val in layer.__dict__.items():
                if isinstance(attr_val, keras.Variable):
                    var_to_owner[id(attr_val)] = (layer, attr_name)
            
            # Also check if variables are in standard Keras lists but maybe not named attributes?
            # Usually Keras variables are attached as attributes (e.g. self.kernel)
            
            # Recurse to sub-layers
            if hasattr(layer, 'layers'):
                stack.extend(layer.layers)
            
            # Handle Functional API / Specific submodules
            for attr_val in layer.__dict__.values():
                if isinstance(attr_val, keras.layers.Layer):
                    stack.append(attr_val)
                elif isinstance(attr_val, (list, tuple)):
                    for item in attr_val:
                        if isinstance(item, keras.layers.Layer):
                            stack.append(item)

        return var_to_owner

    def _replace_variable(self, layer, attr_name, old_var, new_val_tensor):
        """
        Replaces 'old_var' with a new Variable containing 'new_val_tensor' 
        on 'layer' at 'attr_name'. Updates internal weights lists.
        """
        # 1. Create new variable with correct shape/dtype/trainable
        new_var = keras.Variable(
            initializer=new_val_tensor,
            shape=new_val_tensor.shape,
            dtype=old_var.dtype,
            trainable=old_var.trainable,
            name=old_var.name
        )
        
        # 2. Replace attribute on the layer (e.g. layer.kernel = new_var)
        setattr(layer, attr_name, new_var)
        
        # 3. Update Keras internal tracking lists (_trainable_weights, etc.)
        # This ensures model.variables returns the NEW variable.
        for lst_name in ['_trainable_weights', '_non_trainable_weights', '_weights', 'weights', 'variables']:
            if hasattr(layer, lst_name):
                lst = getattr(layer, lst_name)
                if isinstance(lst, list):
                    for i, v in enumerate(lst):
                        if v is old_var:
                            lst[i] = new_var
                            
        return new_var

    def shard_model_parameters(
        self,
        shard_model: "Model",
        weight_loader: Callable[[str], np.ndarray],
        config: Any,
        device_id: Any,
    ) -> Tuple["Model", Set[str]]:
        
        modified = set()
        var_to_owner = self._map_variables_to_owners(shard_model)
        
        for pattern, action in config.state_rules.items():
            if callable(action):
                targets = self._find_matching_parameters(shard_model, pattern)
                
                for name, target_var in targets:
                    # 1. Load ONE weight from disk
                    try:
                        source_val = weight_loader(name)
                        if source_val is None: continue
                    except Exception:
                        continue

                    # 2. Slice (RAM)
                    sliced_val = action(source_val, self.rank)
                    
                    # 3. Create Tensor on Device (Cast to variable dtype, e.g., bfloat16)
                    with keras.device(device_id):
                        sliced_val_tensor = ops.convert_to_tensor(sliced_val, dtype=target_var.dtype)
                    
                    # 4. Assign or Resize
                    if target_var.shape == sliced_val_tensor.shape:
                        target_var.assign(sliced_val_tensor)
                    else:
                        # SHAPE MISMATCH: We must replace the variable
                        if id(target_var) in var_to_owner:
                            layer, attr_name = var_to_owner[id(target_var)]
                            # Perform replacement
                            self._replace_variable(layer, attr_name, target_var, sliced_val_tensor)
                        else:
                            # Fallback: Try assign (will fail, but logs might help debug why owner wasn't found)
                            print(f"⚠️ Warning: Could not find owner for {name} to resize. Attempting direct assign.")
                            target_var.assign(sliced_val_tensor)
                    
                    modified.add(name)
                    
                    # 5. Cleanup
                    del source_val
                    del sliced_val
                    del sliced_val_tensor
        
        gc.collect()
        return shard_model, modified

    def _find_matching_parameters(self, model, pattern: str):
        matches = []
        # Note: We must iterate model.variables dynamically because we might have replaced some!
        # But for 'targets' in one loop, we use the current snapshot.
        for v in model.variables:
            name = v.path if hasattr(v, 'path') else v.name
            if re.search(pattern, name):
                matches.append((name, v))
        return matches

def make_parameter_sharded_model(shard_model, weight_loader, config, rank, device_count, device_id):
    strategy = ParameterShardingStrategy(device_count, rank)
    return strategy.shard_model_parameters(shard_model, weight_loader, config, device_id)