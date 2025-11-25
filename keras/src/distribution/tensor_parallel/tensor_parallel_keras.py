import numpy as np
import keras
from keras import ops
from keras.src.distribution.distribution_lib import list_devices
from keras.src.models import Model
import logging

logger = logging.getLogger(__name__)

# --- Helper Function for Calculating Physical Size ---
def _calculate_physical_shard_size(weight, world_size):
    """
    Calculates the true physical parameter count stored on one device.
    """
    try:
        if not hasattr(weight, '_layout') or weight._layout is None:
            # Replicated variable
            return np.prod(weight.shape)
        
        # Determine which dimension is sharded based on the layout
        # Note: JAX layouts can be complex, this is a simplified check for the 'model' axis
        layout_object = weight._layout
        sharding_factor = 1
        
        # Check if the layout has axis names (Keras TensorLayout)
        if hasattr(layout_object, 'axes'):
            num_model_axes = sum(1 for axis in layout_object.axes if axis == 'model')
            if num_model_axes > 0:
                sharding_factor = world_size
        
        return np.prod(weight.shape) // sharding_factor

    except Exception:
        return 0 


class TensorParallelKeras(Model):
    """
    A Keras Model wrapper for Layout-based Tensor Parallelism.
    
    Since sharding is handled by the backend (JAX Layouts) via AutoTPDistribution,
    this class primarily serves as:
    1. A verification tool (logging sharding status).
    2. A pass-through for compilation and execution.
    3. A context manager for ensuring the DeviceMesh is respected.
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
        self.world_size = world_size or len(list_devices())
        
        # In Layout-based TP, we don't have separate "shard models". 
        # We have ONE model with distributed variables.
        self.distributed = self.world_size > 1
        self.built = True
        
        # Log the status immediately to confirm OOM prevention worked
        if self.distributed:
            self._log_sharding_status()

    def _log_sharding_status(self):
        """Logs the status of variables to confirm sharding occurred."""
        total_logical_params = self._original_model.count_params()
        
        sharded_vars = 0
        total_sharded_params = 0
        total_physical_params_on_one_device = 0
        
        logger.info("-" * 80)
        logger.info(f"SHARDING STATUS VERIFICATION (World Size: {self.world_size})")
        logger.info("-" * 80)
        
        logger.info(f"{'Variable Path':<45} {'Layout Axes':<20} {'Logical Params':<15} {'Physical/Dev':<15}")
        logger.info("-" * 80)
        
        for weight in self._original_model.weights:
            is_sharded = hasattr(weight, '_layout') and weight._layout is not None
            
            # Check if it's effectively replicated (Layout is all None)
            if is_sharded and hasattr(weight._layout, 'axes'):
                 if all(x is None for x in weight._layout.axes):
                     is_sharded = False

            param_count_logical = int(np.prod(weight.shape))
            layout_info = 'REPLICATED'
            param_count_physical = param_count_logical
            
            if is_sharded:
                sharded_vars += 1
                total_sharded_params += param_count_logical
                
                layout_object = weight._layout
                if hasattr(layout_object, 'axes'):
                    layout_info = str(layout_object.axes)
                else:
                    layout_info = 'SHARDED (JAX)'
                
                param_count_physical = _calculate_physical_shard_size(weight, self.world_size)
                
            total_physical_params_on_one_device += param_count_physical
            
            status_prefix = "  ✅" if is_sharded else "  ⚪"
            # Truncate long paths for display
            display_path = (weight.path[:40] + '..') if len(weight.path) > 40 else weight.path
            
            logger.info(
                f"{status_prefix} {display_path:<43}: "
                f"{layout_info:<20} {param_count_logical:,<15} {param_count_physical:,<15}"
            )

        logger.info("-" * 80)
        logger.info(f"Summary:")
        logger.info(f"  Total Sharded Variables: {sharded_vars}")
        logger.info(f"  Total Logical Params:    {total_logical_params:,}")
        logger.info(f"  Params Stored Per GPU:   {total_physical_params_on_one_device:,}")
        
        # Simple verification warning
        if total_physical_params_on_one_device > (total_logical_params * 0.9) and self.world_size > 1:
            logger.warning("⚠️  WARNING: It seems most parameters are REPLICATED. OOM prevention might have failed.")
        else:
            logger.info("✅  SUCCESS: Significant memory reduction verified.")
        logger.info("-" * 80)

    # --- Passthrough Properties ---
    # We must expose the underlying model's variables so Keras can track them
    
    @property
    def variables(self):
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
    
    @property
    def input_spec(self):
        return self._original_model.input_spec

    @property
    def outputs(self):
        return self._original_model.outputs

    @property
    def inputs(self):
        return self._original_model.inputs

    # --- Passthrough Methods ---

    def call(self, inputs, training=None, **kwargs):
        """
        Forward pass.
        JAX automatically handles the communication based on the variable layouts.
        """
        return self._original_model(inputs, training=training, **kwargs)

    def compile(self, optimizer="adam", loss=None, metrics=None, **kwargs):
        """
        Compiles the model. 
        Note: We use the STANDARD optimizer. JAX handles the gradient sharding automatically.
        """
        # We pass the compile call directly to the inner model.
        # The 'distribution' scope (AutoTPDistribution) active in the user script
        # will ensure the optimizer states are created with the correct sharding.
        return self._original_model.compile(
            optimizer=optimizer, 
            loss=loss, 
            metrics=metrics, 
            **kwargs
        )

    def fit(self, x=None, y=None, **kwargs):
        """
        Trains the model.
        Data sharding is handled by AutoTPDistribution.distribute_dataset
        which is triggered automatically by Keras 3 if the distribution scope is active.
        """
        return self._original_model.fit(x, y, **kwargs)

    def evaluate(self, x=None, y=None, **kwargs):
        return self._original_model.evaluate(x, y, **kwargs)

    def predict(self, x, **kwargs):
        return self._original_model.predict(x, **kwargs)
        
    def save(self, filepath, **kwargs):
        # Save the inner model to avoid saving the wrapper
        return self._original_model.save(filepath, **kwargs)
    
    def get_config(self):
        # Enable serialization of the wrapper if needed
        config = super().get_config()
        config.update({
            "world_size": self.world_size,
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Note: Deserializing this wrapper is complex because it expects an instance
        # of a model, not a config. Usually better to rebuild via AutoTPDistribution.
        return super().from_config(config)