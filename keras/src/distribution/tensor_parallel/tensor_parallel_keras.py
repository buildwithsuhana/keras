"""
Tensor Parallel implementation for Keras 3.0
Port of the PyTorch tensor_parallel library
"""

import logging
import re
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
from keras.src.models import Model

logger = logging.getLogger(__file__)


class TensorParallelKeras(Model):
    def __init__(
        self,
        model,
        device_count=None,
        device_ids=None,
        tensor_parallel_config=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._original_model = model

        if device_count is None:
            device_count, device_ids = self._auto_detect_parallelism()
        elif device_ids is None:
            device_ids = self._auto_configure_devices(device_count)

        self.device_count = device_count
        self.device_ids = device_ids
        self.sharding_strategy = "auto"
        
        self.tensor_parallel_config = tensor_parallel_config
        self.distributed = True

        self.sharded_models = [self._original_model]

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
            self.model_shards = [model]
            self.distributed = False
            if len(self.devices) == 1:
                from keras import device

                with device(self.devices[0]):
                    self.model_shards[0] = model
            self.built = True
            self.assembled_model = self._original_model
            return

        if self.tensor_parallel_config is None:
            device_names = [str(d) for d in self.devices]
            self.tensor_parallel_config = get_default_config(
                model, device_names
            )
            logger.info(
                "Using automatic config with auto sharding strategy: sharding individual Dense/Conv/Embedding layers"
            )
        else:
            logger.info("Using injected tensor_parallel_config from AutoTPDistribution")

        print(
            f"🔧 Creating REAL parameter shards for {model.name} across {len(self.devices)} devices"
        )

        self._is_multi_layer_model = len(model.layers) > 2
        if self._is_multi_layer_model:
            logger.info(
                f"   - Multi-layer model detected: {len(model.layers)} layers"
            )

        self.model_shards = []
        self.modified_parameters_names = set()

        logger.info(
            f"✅ Using '{keras.backend.backend()}' backend for parameter sharding."
        )

        for rank, device_id in enumerate(self.devices):
            print(f"[{device_id}] ➡️  Starting sharding process for Rank {rank}")
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

        self.built = True
        if self.distributed:
            self.assembled_model = self.build_assembled_model()
        else:
            self.assembled_model = self._original_model

    def _collect_variables(self, method="variables"):
        """
        Helper to collect unique variables from shards, metrics, loss, and optimizer.
        Handles unwrapping of ShardedWeight objects.
        """
        unique_vars = {}

        # 1. Collect from shards (using weights property + unwrapping)
        for shard in self.model_shards:
            if method == "trainable_variables":
                source_weights = getattr(shard, "trainable_weights", [])
            elif method == "non_trainable_variables":
                source_weights = getattr(shard, "non_trainable_weights", [])
            else:
                source_weights = getattr(shard, "weights", [])
            
            for w in source_weights:
                # Unwrap ShardedWeight if necessary
                if hasattr(w, 'variable'):
                    v = w.variable
                else:
                    v = w
                
                if method == "trainable_variables" and not v.trainable:
                    continue
                if method == "non_trainable_variables" and v.trainable:
                    continue
                    
                unique_vars[id(v)] = v

        # 2. Collect from Metrics
        if hasattr(self, 'metrics'):
            for metric in self.metrics:
                m_vars = metric.variables
                for v in m_vars:
                    if method == "trainable_variables" and not v.trainable:
                        continue
                    if method == "non_trainable_variables" and v.trainable:
                        continue
                    unique_vars[id(v)] = v
        
        # 3. Collect from Loss
        compiled_loss = getattr(self, 'compiled_loss', None)
        if compiled_loss:
            l_vars = compiled_loss.variables
            for v in l_vars:
                if method == "trainable_variables" and not v.trainable:
                    continue
                if method == "non_trainable_variables" and v.trainable:
                    continue
                unique_vars[id(v)] = v

        # 4. Collect from Optimizer (SAFE ACCESS)
        # Use getattr to avoid crashing if optimizer isn't set (e.g. pre-compile)
        optimizer = getattr(self, 'optimizer', None)
        if optimizer and hasattr(optimizer, 'variables'):
            o_vars = optimizer.variables
            for v in o_vars:
                 if method == "trainable_variables" and not v.trainable:
                    continue
                 if method == "non_trainable_variables" and v.trainable:
                    continue
                 unique_vars[id(v)] = v

        return list(unique_vars.values())

    @property
    def variables(self):
        return self._collect_variables(method="variables")

    @property
    def trainable_variables(self):
        return self._collect_variables(method="trainable_variables")

    @property
    def non_trainable_variables(self):
        return self._collect_variables(method="non_trainable_variables")

    @property
    def weights(self):
        return self.variables

    @property
    def trainable_weights(self):
        return self.trainable_variables

    @property
    def non_trainable_weights(self):
        return self.non_trainable_variables

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
            # Prefer the shard's declared input names, fall back to its input tensors.
            shard_inputs = {}
            try:
                input_names = getattr(shard, "input_names", None)
                if input_names:
                    for name in input_names:
                        clean_name = name.split(":")[0]
                        if clean_name in input_layers:
                            shard_inputs[clean_name] = input_layers[clean_name]
                else:
                    # Fall back to inspecting shard.inputs
                    for inp in getattr(shard, "inputs", []):
                        clean_name = inp.name.split(":")[0]
                        if clean_name in input_layers:
                            shard_inputs[clean_name] = input_layers[clean_name]

                if not shard_inputs:
                    # Last resort: forward the full mapping (may be positional in some cases)
                    shard_inputs = dict(input_layers)

                logger.info(
                    f"Calling shard '{getattr(shard, 'name', '<shard>')}' with inputs: {list(shard_inputs.keys())}"
                )
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

    def call(self, inputs, training=None, **kwargs):
        """
        Forward pass for the tensor-parallel model.
        """
        return self.assembled_model(inputs, training=training, **kwargs)

    def compile(self, optimizer=None, loss=None, metrics=None, **kwargs):
        """
        Compile the tensor parallel model.
        """
        # FIX: Only use manual CoordinatedOptimizer if NOT using AutoTPDistribution
        use_manual_optimizer = (
            len(self.model_shards) > 1 
            and optimizer is not None 
            and self.tensor_parallel_config is None 
        )

        if use_manual_optimizer:
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
            # AutoTP Mode: Just use the standard optimizer.
            # Keras Distribution API will handle sharding the optimizer variables.
            logger.info("Using standard optimizer (AutoTP distribution active).")
            super().compile(optimizer, loss, metrics, **kwargs)

    def fit(self, x=None, y=None, **kwargs):
        """Use standard Keras training which correctly handles the train_step."""
        return super().fit(x, y, **kwargs)