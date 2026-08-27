"""
Tensor Parallel implementation for Keras 3.0
Port of the PyTorch tensor_parallel library
"""

import logging
import re
from typing import Optional
from typing import Sequence
from typing import Union

import numpy as np

import keras
from keras import ops
from keras.src.backend import distribution_lib
from keras.src.distribution import list_devices
from keras.src.distribution.tensor_parallel.autoconfig import get_default_config
from keras.src.distribution.tensor_parallel.parameter_sharding import (
    make_parameter_sharded_model,
)

logger = logging.getLogger(__file__)

from keras.src.models import Model


class TensorParallelKeras(Model):
    def __init__(
        self,
        model,
        device_count=None,
        device_ids=None,
        world_size=None,
        device_mesh=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._original_model = model
        self.device_mesh = device_mesh

        if world_size is not None and device_count is None:
            device_count = world_size

        if device_count is None:
            device_count, device_ids = self._auto_detect_parallelism()
        elif device_ids is None:
            device_ids = self._auto_configure_devices(device_count)

        self.device_count = device_count
        self.device_ids = device_ids
        self.sharding_strategy = "auto"

        self.tensor_parallel_config = None
        self.distributed = True

        self.sharded_models = [self._original_model]

        accel_devices = list_devices()
        device_ids = list(self.check_device_ids(device_ids))

        num_processes = distribution_lib.num_processes()

        if accel_devices:
            backend_name = keras.backend.backend()
            print(
                f"🔍 Discovered {len(accel_devices)} devices for backend '{backend_name}'"
            )
            print(f"🔍 Devices: {[str(d) for d in accel_devices]}")

            if num_processes > 1:
                print(
                    f"✅ Multi-process environment detected ({num_processes} processes). Trusting global device_count={device_count}."
                )
                # In multi-process, device_ids should probably stay as requested/configured
            elif len(accel_devices) >= device_count:
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
                "⚠️  Could not discover accelerator devices. Falling back to configuration."
            )

        if not device_ids:
            device_ids = self._auto_configure_devices(device_count)

        if len(device_ids) != device_count:
            device_ids = self._adjust_device_list(device_ids, device_count)

        self.devices = device_ids
        self.device_count = device_count

        if self.device_count <= 1 and num_processes <= 1:
            self.model_shards = [model]
            self.distributed = False
            if len(self.devices) == 1:
                from keras import device

                with device(self.devices[0]):
                    self.model_shards[0] = model

            self.assembled_model = self._original_model

            if hasattr(self._original_model, "inputs"):
                self._inputs = self._original_model.inputs
                self._outputs = self._original_model.outputs

            self.built = True
            return

        if self.tensor_parallel_config is None:
            device_names = [str(d) for d in self.devices]
            
            # Use 'meta' device for analysis if supported by backend
            # to avoid full memory allocation.
            from keras.src import backend
            try:
                with backend.device_scope("meta"):
                    self.tensor_parallel_config = get_default_config(
                        model, device_names
                    )
            except:
                # Fallback for backends that don't support meta device
                self.tensor_parallel_config = get_default_config(
                    model, device_names
                )
            
            print(self.tensor_parallel_config)
            logger.info(
                "Using automatic config with memory-efficient meta-analysis."
            )

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

        process_id = distribution_lib.process_id()

        if num_processes == 1:
            # Single-process mode (e.g. JAX): Shard the original model in-place once.
            shard, modified_names = make_parameter_sharded_model(
                self._original_model,
                self.tensor_parallel_config,
                rank=0,
                device_count=self.device_count,
                device_id=self.devices[0],
                device_mesh=self.device_mesh,
            )
            self.model_shards = [shard]
            self.modified_parameters_names.update(modified_names)
        else:
            # Multi-process mode (e.g. Torch): Each process shards its own copy.
            for rank, device_id in enumerate(self.devices):
                if rank != process_id:
                    continue

                print(f"[{device_id}] ➡️  Starting sharding process for Rank {rank}")
                shard, modified_names = make_parameter_sharded_model(
                    self._original_model,
                    self.tensor_parallel_config,
                    rank=rank,
                    device_count=self.device_count,
                    device_id=device_id,
                    device_mesh=self.device_mesh,
                )
                self.model_shards.append(shard)
                self.modified_parameters_names.update(modified_names)

        self.built = True
        if self.distributed:
            self.assembled_model = self.build_assembled_model()
            self._inputs = self.assembled_model.inputs
            self._outputs = self.assembled_model.outputs
        else:
            self.assembled_model = self._original_model

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
        Returns the original model, which is now patched at the layer level
        to handle collective communications and bias adjustments.
        """
        return self._original_model

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

    def compile(
        self,
        optimizer=None,
        loss=None,
        metrics=None,
        loss_weights=None,
        **kwargs,
    ):
        """
        Compile the tensor parallel model.
        """
        from keras.src.distribution.tensor_parallel.coordinated_optimizer import (
            TensorParallelOptimizer,
        )

        if optimizer is not None and not isinstance(
            optimizer, TensorParallelOptimizer
        ):
            print(
                "🔧 Automatically wrapping optimizer in TensorParallelOptimizer"
            )
            optimizer = TensorParallelOptimizer(
                optimizer,
                device_count=self.device_count,
                tensor_parallel_config=self.tensor_parallel_config,
            )

        super().compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            loss_weights=loss_weights,
            **kwargs,
        )
        logger.info(
            "Compiled TensorParallelKeras model with native Keras distribution logic."
        )

    def summary(
        self,
        line_length=None,
        positions=None,
        print_fn=None,
        expand_nested=False,
        show_trainable=False,
        layer_range=None,
        show_sharding=False,
    ):
        """Prints a string summary of the network."""
        if not show_sharding:
            return super().summary(
                line_length=line_length,
                positions=positions,
                print_fn=print_fn,
                expand_nested=expand_nested,
                show_trainable=show_trainable,
                layer_range=layer_range,
            )

        if print_fn is None:
            print_fn = print

        print_fn("-" * 80)
        print_fn(f'Model: "{self.name}" (Tensor Parallel Sharded)')
        print_fn("-" * 80)
        print_fn(f"{'Variable Path':<50} | {'Sharding Strategy':<20}")
        print_fn("-" * 80)

        sharded_params = set()
        if self.tensor_parallel_config:
            for (
                pattern,
                rule,
            ) in self.tensor_parallel_config.state_rules.items():
                # This is a bit simplified, ideally we match patterns to actual weights
                print_fn(f"{str(pattern):<50} | {str(rule):<20}")
                sharded_params.add(pattern)

        # Also list non-sharded (replicated) parameters
        replicated_count = 0
        for w in self.weights:
            if w.path not in sharded_params and id(w) not in sharded_params:
                replicated_count += 1

        print_fn("-" * 80)
        print_fn(f"Total sharded parameters: {len(sharded_params)}")
        print_fn(f"Total replicated parameters: {replicated_count}")
        print_fn("-" * 80)

    def fit(self, x=None, y=None, **kwargs):
        """Use standard Keras training which correctly handles the train_step."""
        return super().fit(x, y, **kwargs)
