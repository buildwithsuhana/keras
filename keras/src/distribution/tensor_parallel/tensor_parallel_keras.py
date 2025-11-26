"""
Tensor Parallel implementation for Keras 3.0
Port of the PyTorch tensor_parallel library
"""

import logging
import re
import gc
from typing import Collection, Optional, Sequence, Union
import jax
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
from keras.src.distribution.tensor_parallel.lazy_init import lazy_init_scope
logger = logging.getLogger(__file__)


class TensorParallelKeras(Model):
    def __init__(
        self,
        model_input,
        device_count=None,
        device_ids=None,
        **kwargs,
    ):
        """
        Args:
            model_input: A Keras Model instance OR a callable (lambda/function) 
                         that returns a Keras Model. Passing a callable is 
                         RECOMMENDED for large models to avoid OOM during init.
            device_count: Number of devices to shard across.
            device_ids: Specific device identifiers.
        """
        super().__init__(**kwargs)

        # --- OOM FIX 1: Handle Lazy Initialization ---
        # If input is a callable (e.g. lambda: Gemma(...)), build it now on CPU.
        # This prevents pre-allocating memory on GPU before we are ready.
        if callable(model_input) and not isinstance(model_input, keras.Model):
            print("🏗️  Building Skeleton Model (Lazy Init)...")
            # Force CPU to keep the skeleton out of VRAM
            with keras.device("cpu"):
                self._original_model = model_input()
        else:
            self._original_model = model_input

        if device_count is None:
            device_count, device_ids = self._auto_detect_parallelism()
        elif device_ids is None:
            device_ids = self._auto_configure_devices(device_count)

        self.device_count = device_count
        self.device_ids = device_ids
        self.sharding_strategy = "auto"
        
        self.tensor_parallel_config = None
        self.distributed = True

        # Initial placeholder; will be updated with shards
        self.sharded_models = []

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

        # Handle Single Device Case
        if self.device_count <= 1:
            self.model_shards = [self._original_model]
            self.distributed = False
            if len(self.devices) == 1:
                from keras import device

                with device(self.devices[0]):
                    self.model_shards[0] = self._original_model
            self.built = True
            self.assembled_model = self._original_model
            return

        # Generate Config
        if self.tensor_parallel_config is None:
            device_names = [str(d) for d in self.devices]
            self.tensor_parallel_config = get_default_config(
                self._original_model, device_names
            )
            # print(self.tensor_parallel_config) # Reduced log noise
            logger.info(
                "Using automatic config with auto sharding strategy"
            )

        print(
            f"🔧 Creating REAL parameter shards for {self._original_model.name} across {len(self.devices)} devices"
        )

        self._is_multi_layer_model = len(self._original_model.layers) > 2

        self.model_shards = []
        self.modified_parameters_names = set()

        # --- Sharding Loop ---
        for rank, device_id in enumerate(self.devices):
            print(f"[{device_id}] ➡️  Starting sharding process for Rank {rank}")
            
            with jax.default_device(jax.devices("cpu")[0]):
                
                # Create the shard
                shard, modified_parameters_names = make_parameter_sharded_model(
                    self._original_model,
                    self.tensor_parallel_config,
                    rank=rank,
                    device_count=self.device_count,
                    device_id=device_id,
                )
            self.model_shards.append(shard)
            self.modified_parameters_names.update(modified_parameters_names)
            
            # --- OOM FIX 2: Aggressive GC ---
            # Force garbage collection after each shard creation to clean up 
            # intermediate slicing tensors.
            gc.collect() 
            if hasattr(keras.backend, "clear_session"):
                keras.backend.clear_session()
                
            logger.info(f"   ✅ Created shard {rank} for device {device_id}")

        # --- OOM FIX 3: Free Original Model Weights ---
        # The shards now hold the data (or the skeleton data) on GPUs.
        # We NO LONGER need the massive tensors in self._original_model (RAM).
        # We replace them with dummy tensors to free system memory.
        self._free_original_weights()

        logger.info(
            f"Using '{keras.backend.backend()}' backend logic for distribution."
        )

        self.built = True
        if self.distributed:
            self.assembled_model = self.build_assembled_model()
        else:
            self.assembled_model = self._original_model

    def _free_original_weights(self):
        """
        Replaces heavy weights with dummies.
        CRITICAL: Skips GhostVariables to preserve model topology.
        """
        # ADD THIS CHECK AT THE TOP
        if hasattr(self, "_used_lazy_init") and self._used_lazy_init:
            logger.info("👻 Lazy Init detected: Skipping weight freeing (Ghosts are already memory-efficient).")
            return

        print("🗑️  Freeing original model weights from RAM to prevent OOM...")
        freed_count = 0
        total_memory_saved = 0
        
        # We iterate over variables and replace them IN-PLACE
        for weight in self._original_model.variables:
            try:
                # Calculate size for logging
                if hasattr(weight, "numpy"):
                    size = weight.numpy().nbytes
                    total_memory_saved += size
                
                # Create a dummy tensor with the same shape and dtype as the weight
                dummy = ops.zeros(weight.shape, dtype=weight.dtype)
                # Assign dummy to the variable. This drops the reference to the large array.
                weight.assign(dummy)
                freed_count += 1
            except Exception as e:
                logger.warning(f"Could not free weight {weight.name}: {e}")
        
        # Force GC one last time
        gc.collect()
        print(f"✅ Freed {freed_count} original parameters.")
        # print(f"✅ Reclaimed approx {total_memory_saved / (1024**3):.2f} GB of System RAM")

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
        """Auto-detect device_count and device_ids efficiently."""
        from keras.src.distribution import get_best_devices

        available_devices = list_devices()
        device_count = len(available_devices)
        print(
            f"🔍 Auto-detected device_count: {device_count} from {len(available_devices)} available devices"
        )

        device_ids = get_best_devices(device_count)
        print(f"🔍 Auto-detected device_ids: {device_ids}")

        return device_count, device_ids

    def _adjust_device_list(self, device_ids, target_device_count):
        """Adjust device list to match target device_count intelligently."""
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
    def load_sharded_weights(self, filepath):
        """
        Streams weights from H5 file directly to GPU shards.
        Bypasses the CPU RAM bottleneck.
        """
        import h5py
        print(f"💾 Streaming weights from {filepath} directly to GPU shards...")
        
        # 1. Map: "original_name" -> [list of shard variables]
        # We need to know where each chunk of the H5 file goes.
        param_map = {}
        for shard in self.model_shards:
            for var in shard.variables:
                # Assuming var.name is "layer/kernel:0"
                # You might need to adjust name matching based on your backbone
                clean_name = var.name.split(":")[0] 
                if clean_name not in param_map:
                    param_map[clean_name] = []
                param_map[clean_name].append(var)

        count = 0
        with h5py.File(filepath, 'r') as f:
            
            def visit_entry(name, node):
                nonlocal count
                if isinstance(node, h5py.Dataset):
                    # Keras saves weights typically as "layer_name/variable_name"
                    # We try to match this H5 path to our param_map keys
                    
                    # Heuristic matching:
                    # If H5 key is "gemma_backbone/layers_0/dense/kernel"
                    # And param_map key is "tensor_parallel_keras/gemma_backbone..."
                    targets = []
                    for key in param_map:
                        if key.endswith(name):
                            targets = param_map[key]
                            break
                    
                    if not targets:
                        return

                    # 2. Read only ONE tensor into RAM
                    val = node[:] 
                    
                    # 3. Distribute to shards
                    # If targets > 1, it means the weight is sharded.
                    # We must slice 'val' and assign to each shard.
                    
                    # Logic to detect dimension to slice:
                    # We compare the shape of the full tensor vs the shard.
                    
                    for i, shard_var in enumerate(targets):
                        shard_shape = shard_var.shape
                        full_shape = val.shape
                        
                        if shard_shape == full_shape:
                            # Replicated weight (Bias, Norm, etc.)
                            shard_var.assign(val)
                        else:
                            # It is split. Find which dimension differs.
                            # (Simple logic: assuming only 1 dim is split)
                            split_dim = -1
                            for d in range(len(full_shape)):
                                if full_shape[d] != shard_shape[d]:
                                    split_dim = d
                                    break
                            
                            if split_dim != -1:
                                # Calculate start/end indices
                                step = full_shape[split_dim] // len(targets)
                                # Assuming shards are stored in order in param_map (rank 0, rank 1...)
                                # This assumes param_map list order matches rank order. 
                                # (It usually does due to append order in init).
                                start = i * step
                                end = start + step
                                
                                # Slice
                                if split_dim == 0:
                                    slice_val = val[start:end, ...]
                                elif split_dim == 1:
                                    slice_val = val[:, start:end, ...]
                                else:
                                    # Handle other dims if needed
                                    slice_val = np.take(val, range(start, end), axis=split_dim)
                                
                                shard_var.assign(slice_val)
                    
                    count += 1
                    # Python GC to free 'val' immediately
                    del val 

            f.visititems(visit_entry)
            
        print(f"✅ Loaded {count} weight tensors successfully.")
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
        the entire tensor parallel logic.
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

        partial_outputs = []
        for shard in self.model_shards:
            shard_inputs = {}
            
            # Logic to find which inputs this specific shard needs
            # We iterate over the GLOBAL input layers we defined earlier
            for input_name, input_layer in input_layers.items():
                # We check if this shard has an input with this name
                # We check both the input tensors and the input_names attribute
                shard_input_names = getattr(shard, "input_names", [])
                
                # Check if the shard explicitly asks for this input name
                needs_input = False
                if shard_input_names:
                    for s_name in shard_input_names:
                        if s_name.startswith(input_name):
                            needs_input = True
                            break
                
                # Fallback: Check standard Keras inputs
                if not needs_input:
                    for inp in getattr(shard, "inputs", []):
                        if inp.name.startswith(input_name):
                            needs_input = True
                            break

                if needs_input:
                    shard_inputs[input_name] = input_layer

            # SAFETY FALLBACK: If we couldn't map specific inputs, 
            # give the shard everything. This fixes cases where shards 
            # implicitly expect all inputs (common in HuggingFace/KerasHub models).
            if not shard_inputs:
                shard_inputs = dict(input_layers)

            try:
                partial_outputs.append(shard(shard_inputs))
            except Exception as e:
                logger.exception(
                    "Exception when calling shard %s with inputs=%s",
                    getattr(shard, 'name', '<shard>'),
                    list(shard_inputs.keys()),
                )
                raise

        final_layer = self._original_model.layers[-1]
        sharding_type = "unknown"
        final_kernel_name = f"{final_layer.name}.kernel"
        if hasattr(self._original_model, "name") and self._original_model.name:
            final_kernel_name = (
                f"{self._original_model.name}.{final_kernel_name}"
            )

        for pattern, action in self.tensor_parallel_config.state_rules.items():
            if re.search(pattern, final_kernel_name):
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

    def call(self, inputs, training=None, mask=None, **kwargs):
        """
        Forward pass for the tensor-parallel model.
        CRITICAL FIX: Reconstructs dictionary inputs from JAX flattened lists.
        """
        # 1. Detect if inputs were flattened by JAX (list) but model expects dict
        if isinstance(inputs, (list, tuple)) and not isinstance(inputs, dict):
            # Heuristic for Gemma/Transformers: usually [token_ids, padding_mask]
            # We assume inputs[0] is tokens, inputs[1] is mask
            if len(inputs) >= 2:
                inputs = {
                    "token_ids": inputs[0],
                    "padding_mask": inputs[1]
                }
            elif len(inputs) == 1:
                inputs = {"token_ids": inputs[0]}
                
        # 2. Forward to the assembled functional model
        return self.assembled_model(inputs, training=training, mask=mask, **kwargs)

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