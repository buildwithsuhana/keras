import re
import gc
import keras
import jax

class ParameterShardingStrategy:
    def __init__(self, device_count: int, rank: int):
        self.device_count = device_count
        self.rank = rank

    def _replace_variable(self, layer, attr_name, old_var, new_val_tensor, device_id=None):
        # Create a new Variable strictly anchored to the specific GPU
        new_name = f"{old_var.name.split(':')[0]}_shard_{self.rank}"
        
        with keras.device(device_id):
            new_var = keras.Variable(
                initializer=new_val_tensor,
                shape=new_val_tensor.shape,
                dtype=old_var.dtype,
                trainable=old_var.trainable,
                name=new_name 
            )

        # Update the layer attribute
        object.__setattr__(layer, attr_name, new_var)

        # FIXED: Update internal Keras tracking lists so model.trainable_variables
        # points to the new GPU Variable instead of the old CPU one.
        for attr in ['_trainable_weights', '_non_trainable_weights', '_weights']:
            if hasattr(layer, attr):
                weights_list = getattr(layer, attr)
                if isinstance(weights_list, list):
                    for i, v in enumerate(weights_list):
                        if v is old_var:
                            weights_list[i] = new_var
        return new_var

    def shard_model_parameters(self, shard_model, weight_loader, config, device_id):
        modified = set()
        var_to_owner = {}
        for layer in shard_model._flatten_layers(include_self=True):
            for attr in dir(layer):
                val = getattr(layer, attr, None)
                if isinstance(val, keras.Variable):
                    var_to_owner[id(val)] = (layer, attr)

        # Resolve the actual hardware device for JAX
        idx = int(device_id.split(":")[-1]) if ":" in device_id else 0
        jax_target = jax.devices('gpu')[idx]

        for pattern, action in config.state_rules.items():
            if not callable(action): continue
            
            for v in shard_model.variables:
                v_path = v.path if hasattr(v, 'path') else v.name
                if not re.search(pattern, v_path): continue
                
                source_val = weight_loader(v_path)
                if source_val is None: continue

                # Slice on CPU, then transfer directly to the target GPU
                sliced_val = action(source_val, self.rank)
                sliced_val_tensor = jax.device_put(sliced_val, jax_target).astype(v.dtype)

                layer, attr_name = var_to_owner.get(id(v), (None, None))
                if layer and attr_name:
                    self._replace_variable(layer, attr_name, v, sliced_val_tensor, device_id=device_id)
                    modified.add(v_path)
                
                # EFFICIENCY FIX: Prevent memory spikes by waiting for the async transfer 
                # to finish before the next allocation.
                sliced_val_tensor.block_until_ready()
                del source_val, sliced_val, sliced_val_tensor
                gc.collect()
        
        return shard_model, modified

def make_parameter_sharded_model(shard_model, weight_loader, config, rank, device_count, device_id):
    strategy = ParameterShardingStrategy(device_count, rank)
    return strategy.shard_model_parameters(shard_model, weight_loader, config, device_id)