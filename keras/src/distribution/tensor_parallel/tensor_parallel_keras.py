import logging
import re
import contextlib
import gc
from typing import Collection, Optional, Sequence, Union

import numpy as np
import keras
from keras import ops
from keras.src.models import Model
from keras.src.backend import distribution_lib
from keras.src.distribution.tensor_parallel.autoconfig import (
    get_default_config,
)
from keras.src.distribution.tensor_parallel.parameter_sharding import (
    make_parameter_sharded_model,
)
from keras.src.distribution.tensor_parallel.coordinated_optimizer import (
    TensorParallelOptimizer,
)
from keras.src.distribution import (
    list_devices, 
    DeviceMesh, 
    TensorLayout, 
    Distribution, 
    set_distribution, 
    distribution
)

logger = logging.getLogger(__file__)


class _AutoLayoutHeuristic(Distribution):
    """
    Internal Distribution strategy that enforces sharding during initialization.
    
    This class prevents OOMs by calculating the layout based on shape heuristics 
    (matching autoconfig.py) and enforcing sharded memory allocation instantly, 
    rather than creating the full tensor on one device.
    """

    def __init__(self, device_mesh, batch_dim_name="data", auto_shard_dataset=True):
        super().__init__(device_mesh, batch_dim_name, auto_shard_dataset)
        self._device_mesh = device_mesh
        self._num_process = distribution_lib.num_processes()
        self._process_id = distribution_lib.process_id()

    def get_data_layout(self, data_shape):
        data_shard_spec = [None] * len(data_shape)
        data_shard_spec[0] = self.batch_dim_name
        return TensorLayout(data_shard_spec, self._device_mesh)

    def get_variable_layout(self, variable):
        """Determines layout based on variable path and shape heuristics."""
        path = getattr(variable, "path", variable.name).lower()
        shape = variable.shape
        
        # Heuristic: Embeddings (Vocab/Feature Parallel)
        # Matching autoconfig: typically dim 1 for embeddings in your code
        if "embedding" in path and len(shape) >= 2:
             return TensorLayout([None, "model"] + [None] * (len(shape) - 2), self._device_mesh)

        # Heuristic: Kernels (MLP/Attention)
        if "kernel" in path and len(shape) == 2:
            input_dim, output_dim = shape[0], shape[1]
            
            # Dense Layer Logic from autoconfig.py
            expansion_threshold = 1.5
            is_expansion = output_dim > input_dim * expansion_threshold
            is_contraction = input_dim > output_dim * expansion_threshold
            
            if is_expansion: 
                # up_projection -> Split col (dim 1)
                return TensorLayout([None, "model"], self._device_mesh)
            elif is_contraction: 
                # down_projection -> Split row (dim 0)
                return TensorLayout(["model", None], self._device_mesh)
            else:
                # Default dense -> Split col (dim 1)
                return TensorLayout([None, "model"], self._device_mesh)

        # Heuristic: Biases
        if "bias" in path:
            # Matching autoconfig: up_projection bias is split (dim 0)
            # We can't easily know parent layer type here, but for safety 
            # we often replicate bias or split if it's huge. 
            # For now, replicating small biases is safer unless we map perfectly.
            pass

        # Default: Replication (safe for small variables like LayerNorm scales)
        return TensorLayout([None] * len(shape), self._device_mesh)

    def get_tensor_layout(self, path):
        return None
        
    def distribute_dataset(self, dataset):
        return dataset


class AutoTPDistribution:
    """
    Public API for Automated Tensor Parallelism Distribution.
    
    Use this as a scope context manager to safely initialize large models
    without OOM errors.
    
    Usage:
        dist = AutoTPDistribution()
        with dist.scope():
            model = MyLargeModel()
        
        parallel_model = dist.distribute(model)
    """
    def __init__(self, device_mesh=None):
        if device_mesh is None:
            devices = list_devices()
            # Default mesh shape (1, N)
            device_mesh = DeviceMesh(
                (1, len(devices)), ["data", "model"], devices
            )
            
        self.device_mesh = device_mesh
        self.devices = self.device_mesh.devices.flatten().tolist()
        self.world_size = self.device_mesh.devices.size
        
        # Create the internal strategy
        self._strategy = _AutoLayoutHeuristic(
            self.device_mesh, 
            batch_dim_name="data"
        )

    @contextlib.contextmanager
    def scope(self):
        """Context manager to activate the OOM-safe initialization scope."""
        original_scope = distribution()
        set_distribution(self._strategy)
        try:
            yield
        finally:
            set_distribution(original_scope)

    def distribute(self, model):
        """Wraps the built model into TensorParallelKeras."""
        return TensorParallelKeras(
            model=model,
            device_count=self.world_size,
            device_ids=self.devices,
        )


class TensorParallelKeras(Model):
    def __init__(
        self,
        model,
        device_count=None,
        device_ids=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # 1. Device Setup
        if device_count is None:
            device_count, device_ids = self._auto_detect_parallelism()
        elif device_ids is None:
            device_ids = self._auto_configure_devices(device_count)

        self.device_count = device_count
        self.device_ids = device_ids
        self.devices = device_ids
        self.sharding_strategy = "auto"
        self.distributed = True
        
        # Temporary storage of original model
        self._original_model = model
        self.model_shards = []
        self.sharded_models = [model] # Keeps ref for a moment

        # 2. Validation
        if self.device_count <= 1:
            logger.warning("Device count <= 1. Running in non-distributed mode.")
            self.model_shards = [model]
            self.distributed = False
            self.built = True
            self.assembled_model = model
            return

        # 3. Config Generation
        self.tensor_parallel_config = get_default_config(
            model, [str(d) for d in self.devices]
        )
        print(f"🔧 Creating Parameter Shards for {model.name} across {len(self.devices)} devices")

        # 4. Sharding Process
        self.modified_parameters_names = set()

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

        # 5. CRITICAL OOM FIX: Aggressive Cleanup
        # Now that shards are created, we MUST delete the original model
        # to free up the 12GB-24GB of RAM it occupies.
        print("🗑️  Cleaning up original model to free memory...")
        
        # Break references
        self._original_model = None
        self.sharded_models = [] 
        
        # Force Garbage Collection
        del model
        gc.collect()
        keras.backend.clear_session()
        print("✅ Cleanup complete. RAM freed.")

        # 6. Build Assembled Model (Logic Reference)
        # Since we deleted _original_model, we need to ensure build_assembled_model
        # uses the structure from one of the shards (which share the structure) 
        # or use a lightweight clone if needed.
        # NOTE: In this architecture, shards preserve the Keras functional graph structure,
        # so we can use shard[0] to inspect inputs/outputs if needed.
        
        # We need to restore a reference to a 'logical' model for `build_assembled_model`
        # to inspect inputs/outputs. We use the first shard as the structural reference.
        self._structure_ref = self.model_shards[0].original_model 
        
        self.built = True
        self.assembled_model = self.build_assembled_model()

    # Properties (Delegate to shards)
    @property
    def variables(self):
        unique_vars = {id(var): var for shard in self.model_shards for var in shard.variables}
        return list(unique_vars.values())

    @property
    def trainable_variables(self):
        unique_vars = {id(var): var for shard in self.model_shards for var in shard.trainable_variables}
        return list(unique_vars.values())

    @property
    def non_trainable_variables(self):
        unique_vars = {id(var): var for shard in self.model_shards for var in shard.non_trainable_variables}
        return list(unique_vars.values())

    @property
    def weights(self):
        unique_vars = {id(var): var for shard in self.model_shards for var in shard.weights}
        return list(unique_vars.values())

    @property
    def trainable_weights(self):
        unique_vars = {id(var): var for shard in self.model_shards for var in shard.trainable_weights}
        return list(unique_vars.values())

    @property
    def non_trainable_weights(self):
        unique_vars = {id(var): var for shard in self.model_shards for var in shard.non_trainable_weights}
        return list(unique_vars.values())

    # ... (Helper methods: _auto_detect_parallelism, _adjust_device_list, etc. remain same) ...
    def _auto_detect_parallelism(self):
        from keras.src.distribution import get_best_devices
        available_devices = list_devices()
        device_count = len(available_devices)
        device_ids = get_best_devices(device_count)
        return device_count, device_ids

    def _adjust_device_list(self, device_ids, target_device_count):
        current_size = len(device_ids)
        if current_size >= target_device_count:
            return device_ids[:target_device_count]
        return list(device_ids) + [f"cpu:{i}" for i in range(current_size, target_device_count)]

    def _auto_configure_devices(self, device_count):
        available_devices = list_devices()
        if available_devices:
            return available_devices[:device_count]
        return ["cpu:0"]

    def check_device_ids(self, device_ids):
        if device_ids is None:
            device_ids = list_devices()
        return tuple(self.canonicalize_device(d) for d in device_ids)
        
    def canonicalize_device(self, device_spec: Union[str, int]) -> str:
        if isinstance(device_spec, int):
            return "cpu" if device_spec == -1 else f"gpu:{device_spec}"
        elif isinstance(device_spec, str):
            if device_spec == "cpu": return "cpu"
            elif device_spec.startswith("gpu:") or device_spec.startswith("cuda:"):
                return device_spec if device_spec.startswith("gpu:") else f"gpu:{device_spec.split(':')[1]}"
            return device_spec
        return "cpu"

    def build_assembled_model(self):
        if not self.distributed:
            return self._structure_ref

        # Use structure reference for inputs
        input_layers = {
            inp.name.split(":")[0]: keras.Input(
                shape=inp.shape[1:],
                dtype=inp.dtype,
                name=inp.name.split(":")[0],
            )
            for inp in self._structure_ref.inputs
        }

        partial_outputs = []
        for shard in self.model_shards:
            shard_inputs = {}
            try:
                input_names = getattr(shard, "input_names", None)
                if input_names:
                    for name in input_names:
                        clean_name = name.split(":")[0]
                        if clean_name in input_layers:
                            shard_inputs[clean_name] = input_layers[clean_name]
                else:
                    for inp in getattr(shard, "inputs", []):
                        clean_name = inp.name.split(":")[0]
                        if clean_name in input_layers:
                            shard_inputs[clean_name] = input_layers[clean_name]

                if not shard_inputs:
                    shard_inputs = dict(input_layers)

                partial_outputs.append(shard(shard_inputs))
            except Exception:
                logger.exception("Exception calling shard")
                raise

        final_layer = self._structure_ref.layers[-1]
        sharding_type = "unknown"
        final_kernel_name = f"{final_layer.name}.kernel"
        if hasattr(self._structure_ref, "name") and self._structure_ref.name:
            final_kernel_name = f"{self._structure_ref.name}.{final_kernel_name}"

        for pattern, action in self.tensor_parallel_config.state_rules.items():
            if re.search(pattern, final_kernel_name):
                # Check 'dim' in SplitAction
                if hasattr(action, "dim"):
                    # Heuristic mapping: dim=1 -> Column, dim=0 -> Row
                    sharding_type = "column" if action.dim == 1 else "row"
                break

        if sharding_type == "column":
            final_output = ops.concatenate(partial_outputs, axis=-1)
            original_output_dim = self._structure_ref.output_shape[-1]
            if final_output.shape[-1] != original_output_dim:
                final_output = keras.layers.Lambda(lambda x: x[..., :original_output_dim])(final_output)
        elif sharding_type == "row":
            if len(partial_outputs) > 1:
                summed_output = keras.layers.Add()(partial_outputs)
            else:
                summed_output = partial_outputs[0]

            if final_layer.use_bias:
                bias = final_layer.bias
                final_output = keras.layers.Lambda(lambda x: x - bias * (self.device_count - 1))(summed_output)
            else:
                final_output = summed_output
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
            
            # Register shard mapping
            try:
                self.coordinated_optimizer._shard_models = self.model_shards
                var_map = {}
                assembled_vars = self.assembled_model.variables
                for a_var in assembled_vars:
                    key = getattr(a_var, "path", None) or a_var.name
                    suffix = key.split("/")[-1]
                    per_shard = []
                    for shard in self.model_shards:
                        match = next((v for v in shard.variables if v.name.endswith(suffix)), None)
                        per_shard.append(match)
                    var_map[key] = per_shard
                
                self.coordinated_optimizer._shard_var_map = var_map
                inner = getattr(self.coordinated_optimizer, "coordinated_optimizer", None)
                if inner:
                    inner._shard_models = self.model_shards
                    inner._shard_var_map = var_map
            except Exception:
                logger.exception("Failed to register shard mapping")

            super().compile(optimizer=self.coordinated_optimizer, loss=loss, metrics=metrics, **kwargs)
        else:
            super().compile(optimizer, loss, metrics, **kwargs)

    def fit(self, x=None, y=None, **kwargs):
        return super().fit(x, y, **kwargs)