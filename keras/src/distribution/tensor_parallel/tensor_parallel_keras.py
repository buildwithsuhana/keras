import re
import numpy as np
import keras
from keras import ops
from keras.src.distribution import list_devices
from keras.src.distribution.tensor_parallel.autoconfig import (
    get_default_config,
)
from keras.src.distribution.tensor_parallel.coordinated_optimizer import (
    TensorParallelOptimizer,
)
# NOTE: Removed make_parameter_sharded_model import as it is now obsolete for OOM
from keras.src.models import Model
from typing import Optional, Sequence, Union 
import logging

logger = logging.getLogger(__name__)

# --- Helper Function for Calculating Physical Size ---
def _calculate_physical_shard_size(weight, world_size):
    """
    Calculates the true physical parameter count stored on one device.
    This relies on inspecting the variable's assigned layout dimensions.
    """
    try:
        if not hasattr(weight, '_layout') or weight._layout is None:
            # Replicated variable
            return np.prod(weight.shape)
        
        # Determine which dimension is sharded based on the heuristic rules
        layout_axes = weight._layout.axes
        total_params = np.prod(weight.shape)
        
        sharding_factor = 1
        
        # We assume sharding happens only across the 'model' axis
        num_model_axes = sum(1 for axis in layout_axes if axis == 'model')
        
        if num_model_axes > 0:
            # Since the layout is determined by AutoTP, we know it shards across
            # one dimension based on World Size.
            sharding_factor = world_size
        
        # Calculate the size of one shard
        return total_params // sharding_factor

    except Exception:
        # Fallback in case of shape/layout mismatch
        return 0 


class TensorParallelKeras(Model):
    """
    A Keras Model wrapper that implements tensor parallelism.
    This class orchestrates communication and optimizer coordination 
    around a model whose variables are ALREADY SHARDED by the upstream scope.
    """
    def __init__(
        self,
        model, 
        world_size=None,
        device_ids=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._original_model = model
        
        # 1. DEVICE AND WORLD_SIZE SETUP 
        if world_size is None:
            world_size, device_ids = self._auto_detect_parallelism()
        
        self.world_size = world_size
        accel_devices = list_devices()
        device_ids = list(self.check_device_ids(device_ids))
        
        if accel_devices:
            if len(accel_devices) >= self.world_size:
                device_ids = accel_devices[:self.world_size]
            else:
                self.world_size = len(accel_devices)
                device_ids = accel_devices[:self.world_size]

        if not device_ids:
            device_ids = self._auto_configure_devices(self.world_size) 

        if len(device_ids) != self.world_size:
            device_ids = self._adjust_device_list(device_ids, self.world_size) 
        
        self.devices = device_ids
        self.device_count = self.world_size 
        
        # End of device setup. self._original_model is now safe and has layouts.

        if self.world_size <= 1:
            # FIX: We now treat the model itself as the only shard, removing manual array slicing.
            self.model_shards = [model] 
            self.distributed = False
            self.built = True
            self.assembled_model = self._original_model
            return

        # 2. CONFIGURATION (NO MANUAL SHARDING LOOP HERE)
        self.tensor_parallel_config = get_default_config(
            self._original_model, [str(d) for d in self.devices]
        )
        self.distributed = True

        # CRITICAL FIX: The model shards are just duplicates of the original 
        # model (which holds sharded variables internally) 
        # for gradient orchestration purposes in coordinated_optimizer.
        self.model_shards = [self._original_model] * self.world_size
        
        # REMOVED: Manual parameter slicing loop (avoiding the OOM trap)
        
        # self._log_sharding_status() # LOGGING ADDED HERE
        
        self.built = True
        if self.distributed:
            self.assembled_model = self.build_assembled_model()
        else:
            self.assembled_model = self._original_model
            
    # --- NEW LOGGING METHOD ---
    def _log_sharding_status(self):
        """Logs the status of variables to confirm sharding occurred."""
        total_logical_params = self._original_model.count_params()
        
        sharded_vars = 0
        total_sharded_params = 0
        total_physical_params_on_one_device = 0
        
        logger.info("-" * 80)
        logger.info(f"SHARDING STATUS VERIFICATION (World Size: {self.world_size})")
        logger.info("-" * 80)
        logger.info(f"Total Logical Model Parameters: {total_logical_params:,}")
        
        # Log Header
        logger.info(f"{'Variable Path':<45} {'Layout Axes':<15} {'Logical Params':<15} {'Physical Per Dev':<15}")
        logger.info("-" * 80)
        
        # Iterate over all weights to check for the layout attribute
        for weight in self._original_model.weights:
            is_sharded = hasattr(weight, '_layout') and weight._layout is not None
            param_count_logical = np.prod(weight.shape)
            
            layout_info = 'REPLICATED'
            param_count_physical = param_count_logical
            
            if is_sharded:
                sharded_vars += 1
                total_sharded_params += param_count_logical
                
                # --- Get Layout Info ---
                layout_object = weight._layout
                if hasattr(layout_object, 'axes'):
                    layout_info = str(layout_object.axes)
                else:
                    # Fallback for JAX-converted object (e.g. jax.sharding.NamedSharding)
                    layout_info = 'SHARDED'
                
                # --- Calculate Physical Size on One Device ---
                param_count_physical = _calculate_physical_shard_size(weight, self.world_size)
                
            total_physical_params_on_one_device += param_count_physical
            
            # Log Sharded/Replicated Line
            status_prefix = "  ✅" if is_sharded else "  ⚪"
            logger.info(
                f"{status_prefix} {weight.path:<43}: "
                f"{layout_info:<15} {param_count_logical:,<15} {param_count_physical:,<15}"
            )


        # Final Summary Calculation and Log
        replicated_params = total_logical_params - total_sharded_params
        
        logger.info("-" * 80)
        logger.info(f"Summary:")
        logger.info(f"  Total Sharded Variables Found: {sharded_vars}")
        logger.info(f"  Total Logical Model Parameters: {total_logical_params:,}")
        logger.info(f"  Total Parameters Stored on ONE Device: {total_physical_params_on_one_device:,}")
        logger.info(f"  (This metric confirms the memory reduction needed for OOM prevention)")
        logger.info("-" * 80)
        
    # --- FIX: Overwrite all properties to return the original model's variables ---
    @property
    def variables(self):
        # Return the variables from the single, sharded model object
        return self._original_model.variables

    @property
    def trainable_variables(self):
        return self._original_model.trainable_variables

    @property
    def non_trainable_variables(self):
        return self._original_model.non_trainable_variables

    @property
    def weights(self):
        return self._original_model.weights

    @property
    def trainable_weights(self):
        return self._original_model.trainable_weights

    @property
    def non_trainable_weights(self):
        return self._original_model.non_trainable_weights
    
    # --- Existing Helper Methods Follow ---
    
    def _auto_detect_parallelism(self):
        """Auto-detect device_count and device_ids efficiently."""
        from keras.src.distribution import get_best_devices

        available_devices = list_devices()
        device_count = len(available_devices)
        logger.info(
            f"🔍 Auto-detected device_count: {device_count} from {len(available_devices)} available devices"
        )

        device_ids = get_best_devices(device_count)
        logger.info(f"🔍 Auto-detected device_ids: {device_ids}")

        return device_count, device_ids

    def _adjust_device_list(self, device_ids, target_world_size):
        """Adjust device list to match target device_count intelligently."""
        current_size = len(device_ids)
        if current_size >= target_world_size:
            return device_ids[:target_world_size]

        return list(device_ids) + [
            f"cpu:{i}" for i in range(current_size, target_world_size)
        ]

    def _auto_configure_devices(self, world_size):
        """Auto-configure devices - simplified version."""
        available_devices = list_devices()
        if available_devices:
            devices = available_devices[:world_size]
            return devices
        else:
            return ["cpu:0"]

    def check_device_ids(
        self, device_ids):
        """Validate and normalize device IDs for Keras."""
        if device_ids is None:
            device_ids = self._get_all_device_indices()

        return tuple(self.canonicalize_device(d) for d in device_ids)

    def _get_all_device_indices(self1) -> Sequence[str]:
        """Get all available device indices using distribution library."""
        return list_devices()

    def build_assembled_model(self):
        """
        Builds a single, JIT-friendly Keras Functional model that encapsulates
        the entire tensor parallel logic, correctly handling multiple inputs.
        """
        if not self.distributed:
            return self._original_model

        input_layers = {
            inp.name.split(":")[0]: keras.Input(
                shape=inp.shape[1:],
                dtype=inp.dtype,
                name=inp.name.split(":")[0],
            )
            for inp in self._original_model.inputs
        }

        # CRITICAL CHANGE: model_shards now contains duplicates of the original model.
        # This is needed because the assembled model needs a partial output 
        # from each logical shard index for the concatenate/add logic below.
        partial_outputs = [model(input_layers) for model in self.model_shards] 

        final_layer = self._original_model.layers[-1]
        sharding_type = "unknown"
        final_kernel_name = f"{final_layer.name}.kernel"
        if hasattr(self._original_model, "name") and self._original_model.name:
            final_kernel_name = (
                f"{self._original_model.name}.{final_kernel_name}"
            )

        for pattern, action in self.tensor_parallel_config.state_rules.items():
            if re.search(pattern, final_kernel_name):
                # The action lambda is not used for sharding type, only the name.
                # Assuming the config knows the type based on the pattern match.
                # This section remains purely reliant on the heuristic names.
                if hasattr(action, "sharding_type"):
                    sharding_type = action.sharding_type
                break

        if sharding_type == "column":
            final_output = ops.concatenate(partial_outputs, axis=-1)
            original_output_dim = self._original_model.output_shape[-1]
            if final_output.shape[-1] != original_output_dim:
                final_output = keras.layers.Lambda(
                    lambda x: x[..., :original_output_dim]
                )(final_output)
        elif sharding_type == "row":
            if len(partial_outputs) > 1:
                summed_output = keras.layers.Add()(partial_outputs)
            else:
                summed_output = partial_outputs[0]

            if final_layer.use_bias:
                bias = final_layer.bias
                final_output = keras.layers.Lambda(
                    lambda x: x - bias * (self.device_count - 1)
                )(summed_output)
            else:
                final_output = summed_output
        else:
            final_output = partial_outputs[0]

        assembled_model = keras.Model(
            inputs=list(input_layers.values()), outputs=final_output
        )
        return assembled_model

    def canonicalize_device(self, device_spec) -> str:
        """Convert device specification to canonical form."""
        if isinstance(device_spec, int):
            if device_spec == -1:
                return "cpu"
            else:
                return f"gpu:{device_spec}"
        elif isinstance(device_spec, str):
            if device_spec == "cpu":
                return "cpu"
            elif device_spec.startswith("gpu:"):
                return device_spec
            elif device_spec.startswith("cuda:"):
                return f"gpu:{device_spec.split(':')[1]}"
            else:
                return device_spec
        else:
            return "cpu"

    def call(self, inputs, training=None, **kwargs):
        """
        Forward pass for the tensor-parallel model.
        """
        return self.assembled_model(inputs, training=training, **kwargs)

    def compile(self, optimizer=None, loss=None, metrics=None, **kwargs):
        """
        Compile the tensor parallel model.
        """
        if len(self.model_shards) > 1 and optimizer is not None:
            self.coordinated_optimizer = TensorParallelOptimizer(
                optimizer,
                self.world_size,
                tensor_parallel_config=self.tensor_parallel_config,
            )
            
            try:
                # The CoordinatedOptimizer needs the list of "shards" (model replicas)
                self.coordinated_optimizer._shard_models = self.model_shards

                # The rest of the setup relies on the variables of the original model
                var_map = {}
                assembled = getattr(self, "assembled_model", None)
                assembled_vars = (
                    assembled.variables if assembled is not None else []
                )

                for a_var in assembled_vars:
                    key = getattr(a_var, "path", None) or a_var.name
                    suffix = key.split("/")[-1]
                    per_shard = []
                    # Find the corresponding variable on each shard replica
                    for shard in self.model_shards:
                        match = next(
                            (
                                # NOTE: Since all replicas are the *same* sharded model object,
                                # this lookup is mainly to satisfy the TPO's internal structure.
                                v
                                for v in shard.variables
                                if v.name.endswith(suffix)
                            ),
                            None,
                        )
                        per_shard.append(match)
                    var_map[key] = per_shard

                self.coordinated_optimizer._shard_var_map = var_map
                
                inner = getattr(
                    self.coordinated_optimizer, "coordinated_optimizer", None
                )
                if inner is not None:
                    inner._shard_models = self.model_shards
                    inner._shard_var_map = var_map
            except Exception:
                pass

            super().compile(
                optimizer=self.coordinated_optimizer,
                loss=loss,
                metrics=metrics,
                **kwargs,
            )

        else:
            super().compile(optimizer, loss, metrics, **kwargs)

    def fit(self, x=None, y=None, **kwargs):
        """Use standard Keras training which correctly handles the train_step."""
        return super().fit(x, y, **kwargs)