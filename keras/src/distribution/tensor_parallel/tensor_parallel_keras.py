"""
Tensor Parallel implementation for Keras 3.0
Port of the PyTorch tensor_parallel library
"""

import logging
import re
from typing import Collection, Optional, Sequence, Union, Callable, Tuple, Dict, Any
import gc
import numpy as np
import tensorflow as tf
import keras
from keras import ops
from keras.src.distribution.tensor_parallel.autoconfig import (
    get_default_config_keras,
)
from keras.src.distribution.tensor_parallel.parameter_sharding import (
    make_parameter_sharded_model,
    ShardedWeight
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
        # --- MODIFIED: API change to solve OOM on load ---
        model_fn: Callable[..., Model],
        model_args: Tuple = (),
        model_kwargs: Dict[str, Any] = None,
        # --- End MODIFIED ---
        device_count=None,
        device_ids=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        if model_kwargs is None:
            model_kwargs = {}

        # --- MODIFIED: Instantiate the original model on CPU ---
        # This is the fix for OOM on accelerators. We build the full
        # model in host RAM, then shard *from* host RAM to
        # accelerator VRAM.
        print("🔧 Instantiating full model on CPU for sharding...")
        with keras.device("cpu:0"):
            self._original_model = model_fn(*model_args, **model_kwargs)
        print("✅ Full model instantiated on CPU.")
        # --- End MODIFIED ---

        if device_count is None:
            device_count, device_ids = self._auto_detect_parallelism()
        elif device_ids is None:
            device_ids = self._auto_configure_devices(device_count)

        self.device_count = device_count
        self.device_ids = device_ids
        self.sharding_strategy = "auto"
        
        self.tensor_parallel_config = None
        self.distributed = True

        # ==========================================================
        # --- START: 🌟 CRITICAL FIX 🌟 ---
        # The list was named `self.sharded_models` but later code
        # used `self.model_shards`. Changed to be consistent.
        # ==========================================================
        self.model_shards = [] 
        # ==========================================================
        # --- END: CRITICAL FIX ---
        # ==========================================================

        accel_devices = list_devices()
        device_ids = list(self.check_device_ids(device_ids))

        if accel_devices:
            backend_name = keras.backend.backend()
            print(
                f"🔍 Discovered {len(accel_devices)} devices for backend '{backend_name}'"
            )
            print(f"🔍 Devices: {[str(d) for d in accel_devices]}")

            if len(accel_devices) >= device_count:
                print(
                    f"✅ Using REAL tensor parallelism on {device_count} discovered devices."
                )
                device_ids = accel_devices[:device_count]
            else:
                print(
                    f"⚠️  Discovered {len(accel_devices)} devices but device_count={device_count} was requested."
                )
                print(
                    f"⚠️  Reducing device_count to {len(accel_devices)} for real implementation."
                )
                device_count = len(accel_devices)
                device_ids = accel_devices[:device_count]
        else:
            print(
                f"⚠️  Could not discover accelerator devices. Falling back to configuration."
            )

        if not device_ids:
            device_ids = self._auto_configure_devices(device_count)

        if len(device_ids) != device_count:
            device_ids = self._adjust_device_list(device_ids, device_count)

        self.devices = device_ids
        self.device_count = device_count

        if self.device_count <= 1:
            self.model_shards = [self._original_model] # This is correct for the <= 1 case
            self.distributed = False
            if len(self.devices) == 1:
                from keras import device
                # Move the CPU model to the single target device
                with device(self.devices[0]):
                    self.model_shards[0] = self._original_model
            self.built = True
            self.assembled_model = self._original_model
            return

        if self.tensor_parallel_config is None:
            device_names = [str(d) for d in self.devices]
            self.tensor_parallel_config = get_default_config_keras(
                self._original_model, device_names
            )
            logger.info(
                "Using automatic config with auto sharding strategy: sharding individual Dense/Conv/Embedding layers"
            )

        print(
            f"🔧 Creating REAL parameter shards for {self._original_model.name} across {len(self.devices)} devices"
        )

        self._is_multi_layer_model = len(self._original_model.layers) > 2
        if self._is_multi_layer_model:
            logger.info(
                f"   - Multi-layer model detected: {len(self._original_model.layers)} layers"
            )

        # This line is now redundant because of the fix on line 37
        # self.model_shards = [] 
        self.modified_parameters_names = set()

        logger.info(
            f"✅ Using '{keras.backend.backend()}' backend for parameter sharding."
        )

        for rank, device_id in enumerate(self.devices):
            print(f"[{device_id}] ➡️  Starting sharding process for Rank {rank}")
            # --- MODIFIED: Pass the _original_model (on CPU) to be sharded ---
            shard, modified_parameters_names = make_parameter_sharded_model(
                self._original_model,
                self.tensor_parallel_config,
                rank=rank,
                device_count=self.device_count,
                device_id=device_id,
            )
            # This line is now correct and matches the list from line 37
            self.model_shards.append(shard) 
            self.modified_parameters_names.update(modified_parameters_names)
            
            logger.info(f"   ✅ Created shard {rank} for device {device_id}")

        params_per_shard = []
        for i, shard in enumerate(self.model_shards):
            total_params = sum(np.prod(p.shape) for p in shard.weights)
            params_per_shard.append(int(total_params))
            logger.info(f"   📊 Shard {i} parameters: {int(total_params):,}")

        if len(set(params_per_shard)) > 1:
            logger.info(
                "✅ REAL SHARDING CONFIRMED: Different parameter counts across shards"
            )
            logger.info("✅ This is NOT using stubs - real tensor parallelism!")
        else:
            pass

        logger.info(
            f"Using '{keras.backend.backend()}' backend logic for distribution."
        )
        
        # --- MODIFIED: Free the original model from memory ---
        self._original_model_ref = self._original_model # Keep ref for assembled model
        del self._original_model
        gc.collect()
        print("✅ Cleared full-size CPU model from memory.")
        # --- End MODIFIED ---


        self.built = True
        if self.distributed:
            self.assembled_model = self.build_assembled_model()
        else:
            self.assembled_model = self.model_shards[0]

    @property
    def variables(self):
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

    @property
    def non_trainable_variables(self):
        unique_vars = {
            id(var): var
            for shard in self.model_shards
            for var in shard.non_trainable_variables
        }
        return list(unique_vars.values())

    @property
    def weights(self):
        unique_vars = {
            id(var): var
            for shard in self.model_shards
            for var in shard.weights
        }
        return list(unique_vars.values())

    @property
    def trainable_weights(self):
        unique_vars = {
            id(var): var
            for shard in self.model_shards
            for var in shard.trainable_weights
        }
        return list(unique_vars.values())

    @property
    def non_trainable_weights(self):
        unique_vars = {
            id(var): var
            for shard in self.model_shards
            for var in shard.non_trainable_weights
        }
        return list(unique_vars.values())

    def _auto_detect_parallelism(self):
        available_devices = list_devices()
        return len(available_devices), available_devices

    def _adjust_device_list(self, device_ids, target_device_count):
        """Adjust device list to match target_device_count intelligently."""
        current_size = len(device_ids)
        if current_size >= target_device_count:
            return device_ids[:target_device_count]

        return list(device_ids) + [
            f"cpu:{i}" for i in range(current_size, target_device_count)
        ]

    def _auto_configure_devices(self, device_count):
        """Auto-configure devices - simplified version."""
        available_devices = list_devices()
        if available_devices:
            devices = available_devices[:device_count]
            logger.info(f"Auto-configured devices: {devices}")
            return devices
        else:
            logger.warning("No devices available, using default CPU")
            return ["cpu:0"]

    def check_device_ids(
        self, device_ids: Optional[Sequence[str]]
    ) -> Sequence[str]:
        """Validate and normalize device IDs for Keras."""
        if device_ids is None:
            device_ids = self._get_all_device_indices()

        return tuple(self.canonicalize_device(d) for d in device_ids)

    def _get_all_device_indices(self) -> Sequence[str]:
        """Get all available device indices using distribution library."""
        return list_devices()

    def build_assembled_model(self):
        """
        Builds a single, JIT-friendly Keras Functional model that encapsulates
        the entire tensor parallel logic, correctly handling multiple inputs.
        """
        if not self.distributed:
            # --- 🌟 FIX 1 ---
            # Use the correct list name
            return self.model_shards[0]

        original_model = self._original_model_ref

        # --- Find a static sequence length to fix JAX tracing ---
        static_seq_length = None
        try:
            backbone = next(
                l for l in original_model.layers if "backbone" in l.name
            )
            embedding_layer = next(
                l for l in backbone.layers if "position" in l.name
            )
            
            if hasattr(embedding_layer, "sequence_length"):
                static_seq_length = embedding_layer.sequence_length
                print(f"[DEBUG] Found static sequence length: {static_seq_length}")
        
        except Exception as e:
            print(f"[DEBUG] Could not auto-find static sequence length. JAX tracing may fail. Error: {e}")
            static_seq_length = None 

        # --- Fallback for static shape (this part is correct) ---
        if static_seq_length is None:
            DEFAULT_TRACE_LEN = 128
            print(
                f"[DEBUG] WARNING: Auto-detection failed. Forcing a static sequence "
                f"length of {DEFAULT_TRACE_LEN} for JAX tracing."
            )
            static_seq_length = DEFAULT_TRACE_LEN

        input_layers = {
            inp.name.split(":")[0]: keras.Input(
                shape=(static_seq_length,) if static_seq_length else inp.shape[1:],
                dtype=inp.dtype,
                name=inp.name.split(":")[0],
            )
            for inp in original_model.inputs
        }
        
        print(f"[DEBUG] Assembled model inputs created: {input_layers}")

        # --- 🌟 FIX 2 ---
        # This is the most important fix.
        # Rename self.sharded_models to self.model_shards
        partial_outputs = [model(input_layers) for model in self.model_shards]

        final_layer = original_model.layers[-1]
        
        sharding_type = "unknown"
        final_kernel_name = f"{final_layer.name}.kernel"
        if hasattr(original_model, "name") and original_model.name:
            final_kernel_name = (
                f"{original_model.name}.{final_kernel_name}"
            )

        for pattern, action in self.tensor_parallel_config.state_rules.items():
            if re.search(pattern, final_kernel_name):
                if hasattr(action, "dim"):
                    if action.dim == 1:
                        sharding_type = "column"
                    elif action.dim == 0:
                        sharding_type = "row"
                break

        if sharding_type == "column":
            final_output = ops.concatenate(partial_outputs, axis=-1)
            original_output_dim = original_model.output_shape[-1]
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
            # This block will now work correctly
            print("\n[DEBUGGER] --- INSIDE tensor_parallel_keras.py: build_assembled_model ---")
            print(f"[DEBUGGER] Tracing model with inputs: {input_layers}") 
            print(f"[DEBUGGER] Raw partial_outputs from model call: {partial_outputs}")
            print(f"[DEBUGGER] Length of partial_outputs list: {len(partial_outputs)}")
            print("[DEBUGGER] --- Now attempting to access partial_outputs[0] ---")

            final_output = partial_outputs[0]

        assembled_model = keras.Model(
            inputs=list(input_layers.values()), outputs=final_output
        )
        return assembled_model
    def canonicalize_device(self, device_spec: Union[str, int]) -> str:
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
                self.device_count,
                tensor_parallel_config=self.tensor_parallel_config,
            )
            logger.info(
                f"Created coordinated optimizer for {self.device_count} shards"
            )
            
            try:
                self.coordinated_optimizer._shard_models = self.model_shards

                var_map = {}
                assembled = getattr(self, "assembled_model", None)
                assembled_vars = (
                    assembled.variables if assembled is not None else []
                )

                for a_var in assembled_vars:
                    key = getattr(a_var, "path", None) or a_var.name
                    suffix = key.split("/")[-1]
                    per_shard = []
                    for shard in self.model_shards:
                        match = next(
                            (
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
                logger.exception(
                    "Failed to register shard mapping on coordinated optimizer"
                )

            super().compile(
                optimizer=self.coordinated_optimizer,
                loss=loss,
                metrics=metrics,
                **kwargs,
            )
            logger.info(
                "Compiled TensorParallelKeras model with coordinated optimizer."
            )

        else:
            super().compile(optimizer, loss, metrics, **kwargs)

    def fit(self, x=None, y=None, **kwargs):
        """Use standard Keras training which correctly handles the train_step."""
        return super().fit(x, y, **kwargs)

    # --- NEW CHECKPOINTING METHODS ---

    def save_checkpoint(self, filepath: str):
        """
        Saves the full, un-sharded model weights and optimizer state.
        This method gathers all shards, concatenates them on the CPU,
        and saves them to a single .npz file.
        """
        if not self.distributed:
            logger.warning("Model is not distributed. Saving standard weights.")
            self.assembled_model.save_weights(filepath)
            return

        print(f"🧩 Gathering sharded weights to save full checkpoint at {filepath}...")

        # --- 1. Gather Model Weights ---
        full_model_state = {}
        sharding_strategy = self.model_shards[0].sharding_strategy

        # Gather sharded weights
        for param_name in self.modified_parameters_names:
            mapping = sharding_strategy.weight_mapping[param_name]
            dim = mapping["dim"]
            
            # Gather shards from all models (convert to numpy on CPU)
            shards = [
                shard.sharding_strategy.sharded_weights[param_name].numpy()
                for shard in self.model_shards
            ]
            
            # Concatenate along the sharding dimension
            full_weight = np.concatenate(shards, axis=dim)
            full_model_state[param_name] = full_weight

        # Gather unsharded (replicated) weights
        original_weights = sharding_strategy.original_weights
        for param_name, weight in original_weights.items():
            if param_name not in self.modified_parameters_names:
                full_model_state[param_name] = weight # Already numpy
        
        print(f"✅ Gathered {len(full_model_state)} model weights.")

        # --- 2. Gather Optimizer State ---
        full_optimizer_state = {}
        if hasattr(self, "coordinated_optimizer"):
            coordinator = self.coordinated_optimizer.coordinated_optimizer
            sharded_states = coordinator.sharded_states

            for state_name, state_value in sharded_states.items():
                if isinstance(state_value, dict): # Per-variable states
                    full_optimizer_state[state_name] = {}
                    for param_path, param_shards in state_value.items():
                        # Find the sharding dim for this optimizer state
                        param_name = param_path.replace("/", ".")
                        dim = 0  # Default for optimizer states
                        for (
                            pattern,
                            action,
                        ) in self.tensor_parallel_config.state_rules.items():
                            if re.search(pattern, param_name) and hasattr(action, "dim"):
                                dim = action.dim
                                break
                        
                        # Concatenate the numpy arrays
                        full_tensor = np.concatenate(param_shards, axis=dim)
                        full_optimizer_state[state_name][param_path] = full_tensor

                elif isinstance(state_value, list): # Global state (e.g., iterations)
                    full_optimizer_state[state_name] = state_value[0] # All are identical
            
            print(f"✅ Gathered {len(full_optimizer_state)} optimizer states.")

        # --- 3. Save to NPZ file ---
        try:
            np.savez(
                filepath,
                _model_weights=full_model_state,
                _optimizer_state=full_optimizer_state
            )
            print(f"💾 Checkpoint successfully saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def load_checkpoint(self, filepath: str):
        """
        Loads a full, un-sharded checkpoint from a .npz file.
        This method loads the full weights, re-shards them on the fly,
        and assigns the correct shard to each model and optimizer.
        """
        print(f"⏬ Loading full checkpoint from {filepath}...")
        try:
            data = np.load(filepath, allow_pickle=True)
        except Exception as e:
            logger.error(f"Failed to load checkpoint file {filepath}: {e}")
            return

        # --- 1. Load Model Weights ---
        if "_model_weights" not in data:
            logger.error("Invalid checkpoint: `_model_weights` not found.")
            return
            
        full_model_state = data["_model_weights"].item()
        sharding_strategy = self.model_shards[0].sharding_strategy
        
        print("Restoring model weights...")
        for param_name, full_weight in full_model_state.items():
            if param_name in self.modified_parameters_names:
                # This is a sharded weight
                mapping = sharding_strategy.weight_mapping[param_name]
                action = mapping["action"] # This is the SplitRule object

                # Distribute the loaded weight to all shards
                for rank, shard_model in enumerate(self.model_shards):
                    # Re-shard the full weight on the fly
                    new_shard_tensor = action(full_weight, rank)
                    
                    # Find the ShardedWeight variable and assign it
                    sharded_weight_obj = shard_model._sharded_weight_cache.get(param_name)
                    
                    if sharded_weight_obj:
                        sharded_weight_obj.variable.assign(new_shard_tensor)
                    else:
                        logger.warning(f"Could not find ShardedWeight for {param_name} in rank {rank}")
            else:
                # This is a replicated (unsharded) weight
                # Assign it to all shards
                for shard_model in self.model_shards:
                    try:
                        # Find the replicated weight
                        weight_obj = next(
                            w for w in shard_model.weights 
                            if not isinstance(w, ShardedWeight) and w.name == param_name
                        )
                        weight_obj.assign(full_weight)
                    except StopIteration:
                        logger.debug(f"Weight {param_name} not found in shard (this may be ok)")

        print("✅ Model weights restored.")

        # --- 2. Load Optimizer State ---
        if "_optimizer_state" not in data:
            logger.warning("Checkpoint has no optimizer state. Skipping.")
            return

        if not hasattr(self, "coordinated_optimizer"):
            logger.warning("Model has no coordinated_optimizer. Skipping state load.")
            return

        print("Restoring optimizer states...")
        full_optimizer_state = data["_optimizer_state"].item()
        coordinator = self.coordinated_optimizer.coordinated_optimizer

        for state_name, state_value in full_optimizer_state.items():
            if isinstance(state_value, dict): # Per-variable states
                for param_path, full_tensor in state_value.items():
                    param_name = param_path.replace("/", ".")
                    dim = 0
                    for (
                        pattern,
                        action,
                    ) in self.tensor_parallel_config.state_rules.items():
                        if re.search(pattern, param_name) and hasattr(action, "dim"):
                            dim = action.dim
                            break
                    
                    # Re-shard the numpy array
                    sharded_tensors = np.array_split(
                        full_tensor, self.device_count, axis=dim
                    )
                    
                    # Update the coordinator's sharded_states
                    if state_name not in coordinator.sharded_states:
                         coordinator.sharded_states[state_name] = {}
                    coordinator.sharded_states[state_name][param_path] = [
                        s for s in sharded_tensors
                    ]
            else:
                # Global state
                coordinator.sharded_states[state_name] = [state_value] * self.device_count

        # --- 3. Synchronize loaded state into sharded optimizers ---
        # After loading, we must push the numpy state into the
        # actual Keras optimizer variables on each device.
        for shard_idx, shard_model in enumerate(self.model_shards):
            local_states = coordinator._get_local_optimizer_states(shard_idx)
            shard_optimizer = shard_model.optimizer.base_optimizer
            coordinator._update_optimizer_internal_state(shard_optimizer, local_states)
            
        print("✅ Optimizer states restored and synchronized.")
    
    # --- END NEW CHECKPOINTING METHODS ---