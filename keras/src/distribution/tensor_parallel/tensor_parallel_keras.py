import logging
import re
import gc
from typing import Optional, Sequence, Union

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
from keras.src.models import Model

logger = logging.getLogger(__file__)

class TensorParallelKeras(Model):
    def __init__(
        self,
        model,
        device_count=None,
        device_ids=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # --- 1. Device Configuration ---
        if device_count is None:
            device_count, device_ids = self._auto_detect_parallelism()
        elif device_ids is None:
            device_ids = self._auto_configure_devices(device_count)

        self.device_count = device_count
        self.devices = list(self.check_device_ids(device_ids))
        
        # Validation
        accel_devices = list_devices()
        if len(accel_devices) < self.device_count:
            print(f"⚠️  Requested {self.device_count} devices, but only found {len(accel_devices)}. Fallback logic may apply.")

        # --- 2. Configuration & Setup ---
        self.tensor_parallel_config = None
        self.distributed = True
        self.model_shards = []

        if self.tensor_parallel_config is None:
            device_names = [str(d) for d in self.devices]
            self.tensor_parallel_config = get_default_config(model, device_names)

        print(f"🔧 Initializing Tensor Parallelism for {model.name} on {len(self.devices)} devices...")

        # --- 3. Iterative Shard Creation (Lazy Init) ---
        # We clone and shard one by one to keep peak memory low.
        
        for rank, device_id in enumerate(self.devices):
            print(f"[{device_id}] ⏳ Creating shard {rank+1}/{self.device_count}...")

            # A. Create the Shard Structure on the Target Device
            # This allocates weights on the GPU immediately (randomly initialized), 
            # avoiding CPU RAM spikes.
            with keras.device(device_id):
                # Clone model structure. 
                # Note: We use a custom object scope if necessary, empty here.
                shard = keras.models.clone_model(model)
                
                # Ensure the shard respects the mixed precision policy
                if hasattr(model, 'dtype_policy'):
                    shard.dtype_policy = model.dtype_policy

            # B. Build the shard to initialize variables (if not already built)
            if not shard.built and model.inputs:
                try:
                    input_shapes = [i.shape for i in model.inputs]
                    shard.build(input_shapes)
                except Exception:
                    pass # Attempt auto-build later if this fails

            # C. Copy & Slice Weights: Source (CPU) -> Shard (GPU)
            # This function reads from 'model', slices, and assigns to 'shard'
            shard, _ = make_parameter_sharded_model(
                shard_model=shard,
                source_model=model,
                config=self.tensor_parallel_config,
                rank=rank,
                device_count=self.device_count,
                device_id=device_id,
            )

            self.model_shards.append(shard)
            
            # D. CRITICAL: Garbage Collection
            # Clean up numpy intermediates generated during slicing
            gc.collect() 
            print(f"[{device_id}] ✅ Shard ready.")

        # --- 4. Cleanup Master Model ---
        # We no longer need the heavy CPU model.
        print("🗑️  Freeing original master model from CPU memory...")
        self._original_model = None # Remove reference
        # Note: We cannot 'del model' here as it is passed by arg, but we drop our ref.
        gc.collect()

        self.built = True
        self.assembled_model = self.build_assembled_model()

    # --- Standard Methods (Keep existing logic mostly, just updated references) ---

    def build_assembled_model(self):
        """Reconstructs the graph for the forward pass across shards."""
        # Use the first shard as reference for structure, but it's logically split
        ref_model = self.model_shards[0]
        
        input_layers = {
            inp.name.split(":")[0]: keras.Input(
                shape=inp.shape[1:],
                dtype=inp.dtype,
                name=inp.name.split(":")[0],
            )
            for inp in ref_model.inputs
        }

        partial_outputs = []
        for shard in self.model_shards:
            # Simple input mapping: assume all inputs go to all shards
            # (In sophisticated TP, inputs might be split too, but we replicate inputs for now)
            shard_inputs = {
                name: input_layers[name.split(":")[0]] 
                for name in [i.name for i in shard.inputs] 
                if name.split(":")[0] in input_layers
            }
            if not shard_inputs:
                shard_inputs = list(input_layers.values()) # Fallback to list
            
            partial_outputs.append(shard(shard_inputs))

        # Output Reduction (Row/Col logic)
        # Simplified reduction for Causal LM (usually just Sum at the end for logits)
        if len(partial_outputs) > 1:
            # For CausalLM head, usually we sum the logits if sharded column-wise
            final_output = keras.layers.Add()(partial_outputs)
        else:
            final_output = partial_outputs[0]

        return keras.Model(inputs=list(input_layers.values()), outputs=final_output)

    def call(self, inputs, training=None, **kwargs):
        return self.assembled_model(inputs, training=training, **kwargs)

    def compile(self, optimizer=None, loss=None, metrics=None, **kwargs):
        if len(self.model_shards) > 1 and optimizer is not None:
            self.coordinated_optimizer = TensorParallelOptimizer(
                optimizer,
                self.device_count,
                tensor_parallel_config=self.tensor_parallel_config,
            )
            # Inject shards into optimizer
            self.coordinated_optimizer._shard_models = self.model_shards
            
            # Create variable mapping for the optimizer
            # This maps master_var_path -> [shard0_var, shard1_var]
            var_map = {}
            for i, shard in enumerate(self.model_shards):
                for v in shard.trainable_variables:
                    # Best effort name matching
                    key = v.path if hasattr(v, "path") else v.name
                    if key not in var_map:
                        var_map[key] = [None] * self.device_count
                    var_map[key][i] = v
            self.coordinated_optimizer._shard_var_map = var_map

            super().compile(
                optimizer=self.coordinated_optimizer,
                loss=loss,
                metrics=metrics,
                **kwargs,
            )
        else:
            super().compile(optimizer, loss, metrics, **kwargs)

    # --- Utils ---
    def _auto_detect_parallelism(self):
        accel = list_devices()
        return len(accel), [str(d) for d in accel]

    def _auto_configure_devices(self, count):
        accel = list_devices()
        return [str(d) for d in accel[:count]]

    def check_device_ids(self, device_ids):
        # normalize
        return [d if "gpu" in d or "cpu" in d or "tpu" in d else f"gpu:{d}" for d in device_ids]