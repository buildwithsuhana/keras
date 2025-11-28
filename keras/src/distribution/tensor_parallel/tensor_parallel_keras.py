"""
Tensor Parallel implementation for Keras 3.0
Port of the PyTorch tensor_parallel library
"""

import logging
import re
import gc
from typing import Collection, Optional, Sequence, Union

import numpy as np
import tensorflow as tf
import keras
from keras import ops
from keras.src.distribution.tensor_parallel.autoconfig import (
    get_default_config,
)
from keras.src.distribution.tensor_parallel.parameter_sharding import (
    make_parameter_sharded_model,
)
from keras.src.distribution.tensor_parallel.coordinated_optimizer import (
    TensorParallelOptimizer,
)

from keras.src.distribution import list_devices

logger = logging.getLogger(__file__)

from keras.src.models import Model


class TensorParallelKeras(Model):
    def __init__(
        self,
        model,
        device_count=None,
        device_ids=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # We assume 'model' is currently on CPU or Meta device.
        # We hold a reference but we DO NOT use it for execution.
        self._original_model = model

        if device_count is None:
            device_count, device_ids = self._auto_detect_parallelism()
        elif device_ids is None:
            device_ids = self._auto_configure_devices(device_count)

        self.device_count = device_count
        self.device_ids = device_ids
        self.sharding_strategy = "auto"
        
        self.tensor_parallel_config = None
        self.distributed = True

        accel_devices = list_devices()
        device_ids = list(self.check_device_ids(device_ids))

        if accel_devices:
            # Device detection logic...
            if len(accel_devices) >= device_count:
                device_ids = accel_devices[:device_count]
            else:
                device_count = len(accel_devices)
                device_ids = accel_devices[:device_count]
        else:
            print(f"⚠️  Could not discover accelerator devices. Falling back to configuration.")

        if not device_ids:
            device_ids = self._auto_configure_devices(device_count)

        if len(device_ids) != device_count:
            device_ids = self._adjust_device_list(device_ids, device_count)

        self.devices = device_ids
        self.device_count = device_count

        # Single device fallback
        if self.device_count <= 1:
            self.model_shards = [model]
            self.distributed = False
            self.built = True
            self.assembled_model = self._original_model
            return

        # Auto Config
        if self.tensor_parallel_config is None:
            device_names = [str(d) for d in self.devices]
            self.tensor_parallel_config = get_default_config(
                model, device_names
            )

        print(
            f"🔧 Creating REAL parameter shards for {model.name} across {len(self.devices)} devices"
        )

        self.model_shards = []
        self.modified_parameters_names = set()

        # SHARDING LOOP
        for rank, device_id in enumerate(self.devices):
            print(f"[{device_id}] ➡️  Starting sharding process for Rank {rank}")
            
            # Force GC before allocating new shard to clear previous shard's CPU debris
            gc.collect()
            
            # Create the shard. 
            # This will Clone the model on CPU, then slice weights to GPU.
            # No full model is ever on GPU.
            shard, modified_parameters_names = make_parameter_sharded_model(
                model,
                self.tensor_parallel_config,
                rank=rank,
                device_count=self.device_count,
                device_id=device_id,
            )
            self.model_shards.append(shard)
            self.modified_parameters_names.update(modified_parameters_names)

            logger.info(f"   ✅ Created shard {rank} for device {device_id}")

        # Final GC to clean up any temporary buffers from slicing
        gc.collect()

        self.built = True
        if self.distributed:
            self.assembled_model = self.build_assembled_model()
        else:
            self.assembled_model = self._original_model

    @property
    def variables(self):
        # Gather all unique variables from all shards
        unique_vars = {
            id(var): var
            for shard in self.model_shards
            for var in shard.variables
        }
        return list(unique_vars.values())

    @property
    def trainable_variables(self):
        unique_vars = {
            id(var): var
            for shard in self.model_shards
            for var in shard.trainable_variables
        }
        return list(unique_vars.values())
    
    # ... (Keep other properties standard) ...

    def _auto_detect_parallelism(self):
        """Auto-detect device_count and device_ids efficiently."""
        from keras.src.distribution import get_best_devices
        available_devices = list_devices()
        device_count = len(available_devices)
        device_ids = get_best_devices(device_count)
        return device_count, device_ids

    def _adjust_device_list(self, device_ids, target_device_count):
        current_size = len(device_ids)
        if current_size >= target_device_count:
            return device_ids[:target_device_count]
        return list(device_ids) + [
            f"cpu:{i}" for i in range(current_size, target_device_count)
        ]

    def _auto_configure_devices(self, device_count):
        available_devices = list_devices()
        if available_devices:
            return available_devices[:device_count]
        return ["cpu:0"]

    def check_device_ids(self, device_ids: Optional[Sequence[str]]) -> Sequence[str]:
        if device_ids is None:
            device_ids = self._get_all_device_indices()
        return tuple(self.canonicalize_device(d) for d in device_ids)

    def _get_all_device_indices(self) -> Sequence[str]:
        return list_devices()

    def build_assembled_model(self):
        if not self.distributed:
            return self._original_model

        # Create input placeholders matching original model
        input_layers = {
            inp.name.split(":")[0]: keras.Input(
                shape=inp.shape[1:],
                dtype=inp.dtype,
                name=inp.name.split(":")[0],
            )
            for inp in self._original_model.inputs
        }

        partial_outputs = []
        for shard in self.model_shards:
            # Map inputs to shard
            # Shard is a ParameterShardedModel which accepts standard inputs
            shard_inputs = dict(input_layers)
            partial_outputs.append(shard(shard_inputs))

        final_layer = self._original_model.layers[-1]
        
        # Determine how to aggregate results based on the last layer's config
        sharding_type = "unknown"
        final_kernel_name = f"{final_layer.name}.kernel"
        if hasattr(self._original_model, "name") and self._original_model.name:
            final_kernel_name = f"{self._original_model.name}.{final_kernel_name}"

        # Look for the last layer in the state rules to guess sharding
        for pattern, action in self.tensor_parallel_config.state_rules.items():
            if re.search(pattern, final_kernel_name):
                 # This logic is a bit heuristic, assuming action has metadata
                 # In updated code, action is lambda. 
                 # We rely on output_rules usually.
                 pass
        
        # Fallback to output rules for the last layer
        output_rule = self.tensor_parallel_config.output_rules.get(final_layer.name, {})
        # If the last layer was already all-reduced/gathered inside the shard,
        # partial_outputs are identical. We just take one.
        
        # If we have multiple outputs and they differ, we might need concatenation
        # But usually TP ends with an AllReduce or Gather.
        
        # Simple aggregation: Average/Sum if they are different shards?
        # If the model ends with a Dense layer that was Column Parallel, output is sharded.
        # If it was Row Parallel, output is AllReduced (replicated).
        
        # We assume the user's config handles the final communication.
        # So we just take the output from the 0-th shard (Rank 0).
        final_output = partial_outputs[0]

        assembled_model = keras.Model(
            inputs=list(input_layers.values()), outputs=final_output
        )
        return assembled_model

    def canonicalize_device(self, device_spec: Union[str, int]) -> str:
        if isinstance(device_spec, int):
            return "cpu" if device_spec == -1 else f"gpu:{device_spec}"
        elif isinstance(device_spec, str):
            if device_spec == "cpu": return "cpu"
            if device_spec.startswith("gpu:") or device_spec.startswith("tpu:"): return device_spec
            if device_spec.startswith("cuda:"): return f"gpu:{device_spec.split(':')[1]}"
            return device_spec
        return "cpu"

    def call(self, inputs, training=None, **kwargs):
        return self.assembled_model(inputs, training=training, **kwargs)

    def compile(self, optimizer=None, loss=None, metrics=None, **kwargs):
        if len(self.model_shards) > 1 and optimizer is not None:
            self.coordinated_optimizer = TensorParallelOptimizer(
                optimizer,
                self.device_count,
                tensor_parallel_config=self.tensor_parallel_config,
            )
            # Register shards
            self.coordinated_optimizer._shard_models = self.model_shards
            
            # Map variables for the optimizer
            var_map = {}
            assembled = getattr(self, "assembled_model", None)
            if assembled:
                for a_var in assembled.variables:
                     key = a_var.path if hasattr(a_var, 'path') else a_var.name
                     suffix = key.split("/")[-1]
                     per_shard = []
                     for shard in self.model_shards:
                         # Find matching var in shard
                         found = None
                         for v in shard.variables:
                             if v.name.endswith(suffix):
                                 found = v
                                 break
                         per_shard.append(found)
                     var_map[key] = per_shard
            
            self.coordinated_optimizer._shard_var_map = var_map
            
            super().compile(
                optimizer=self.coordinated_optimizer,
                loss=loss,
                metrics=metrics,
                **kwargs,
            )
        else:
            super().compile(optimizer, loss, metrics, **kwargs)