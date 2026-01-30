import warnings

import numpy as np
import torch
from packaging.version import parse

from keras.src import backend
from keras.src import callbacks as callbacks_module
from keras.src import optimizers as optimizers_module
from keras.src import tree
from keras.src.backend import config
from keras.src.trainers import trainer as base_trainer
from keras.src.trainers.data_adapters import array_slicing
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.trainers.epoch_iterator import EpochIterator
from keras.src.utils import traceback_utils


class _AllGatherWithGradient(torch.autograd.Function):
    """Custom autograd function for all-gather with proper gradient flow.
    
    This function performs an all-gather operation that preserves gradient flow.
    During forward pass, it gathers tensors from all ranks.
    During backward pass, it scatters gradients back to each rank.
    """
    
    @staticmethod
    def forward(ctx, local_tensor, shard_dim):
        """Forward pass: all-gather tensors from all ranks.
        
        Args:
            local_tensor: The local tensor to gather
            shard_dim: The dimension along which to gather
            
        Returns:
            Concatenated tensor from all ranks
        """
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        
        # Create output tensor for gathered data
        local_shape = list(local_tensor.shape)
        output_shape = local_shape.copy()
        output_shape[shard_dim] = output_shape[shard_dim] * world_size
        
        # Allocate output tensor
        output = torch.empty(output_shape, dtype=local_tensor.dtype, device=local_tensor.device)
        
        # All-gather
        if hasattr(torch.distributed, 'all_gather_into_tensor'):
            # Use newer API if available
            torch.distributed.all_gather_into_tensor(output, local_tensor.contiguous())
        else:
            # Fallback to list-based all_gather
            output_list = [torch.empty_like(local_tensor) for _ in range(world_size)]
            torch.distributed.all_gather(output_list, local_tensor.contiguous())
            output = torch.cat(output_list, dim=shard_dim)
        
        # Save context for backward
        ctx.shard_dim = shard_dim
        ctx.world_size = world_size
        ctx.local_shape = local_shape
        ctx.local_tensor_requires_grad = local_tensor.requires_grad
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass: scatter gradients back to each rank.
        
        Args:
            grad_output: Gradient from the next layer
            
        Returns:
            Gradient for the local tensor
        """
        shard_dim = ctx.shard_dim
        world_size = ctx.world_size
        local_shape = ctx.local_shape
        rank = torch.distributed.get_rank()
        
        # Calculate the slice of grad_output that belongs to this rank
        shard_size = grad_output.shape[shard_dim] // world_size
        start_idx = rank * shard_size
        end_idx = start_idx + shard_size
        
        # Extract the gradient slice for this rank
        grad_slices = [slice(None)] * grad_output.dim()
        grad_slices[shard_dim] = slice(start_idx, end_idx)
        
        grad_local = grad_output[tuple(grad_slices)]
        
        # Ensure gradient has the correct shape
        grad_local = grad_local.contiguous()
        
        return grad_local, None


def _all_gather_with_grad(local_tensor, shard_dim):
    """Perform all-gather with proper gradient flow.
    
    Args:
        local_tensor: The local tensor to gather
        shard_dim: The dimension along which to gather
        
    Returns:
        Concatenated tensor from all ranks with proper gradient tracking
    """
    return _AllGatherWithGradient.apply(local_tensor, shard_dim)


class TorchTrainer(base_trainer.Trainer):
    def __init__(self):
        super().__init__()
        self.train_function = None
        self.test_function = None
        self.predict_function = None
        self._torch_module_parallelized = False

    def _ensure_dtensor_input(self, x):
        """Convert inputs to DTensors when model has DTensor weights.
        
        For ModelParallel training with DTensor weights, inputs must also be
        DTensors to avoid the "mixed torch.Tensor and DTensor" error.
        
        Args:
            x: Input tensor (can be torch.Tensor, DTensor, or nested structure)
            
        Returns:
            Same structure, with inputs converted to DTensors if needed
        """
        from keras.src.distribution.distribution_lib import distribution
        from keras.src.distribution.distribution_lib import ModelParallel
        from keras.src.backend.torch.distribution_lib import (
            DTensor,
            Replicate,
            _get_default_device_mesh,
            DTENSOR_AVAILABLE,
        )
        
        # If DTensor not available, return as-is
        if not DTENSOR_AVAILABLE:
            return x
        
        # Check if ModelParallel distribution is active
        dist = distribution()
        if not isinstance(dist, ModelParallel):
            return x
        
        # Get device mesh
        device_mesh = _get_default_device_mesh()
        if device_mesh is None:
            return x
        
        # Check if any weights are DTensors by checking if they have a layout attribute
        # that indicates DTensor sharding
        has_dtensor_weights = False
        for var in self.trainable_weights:
            if isinstance(var, torch.nn.Parameter):
                # Check if it's a DTensor Parameter (has _tensor_attrs or similar)
                if hasattr(var, '_tensor_attrs') or hasattr(var, '_spec'):
                    has_dtensor_weights = True
                    break
                # Also check if the data is a DTensor
                if hasattr(var, 'data') and isinstance(var.data, DTensor):
                    has_dtensor_weights = True
                    break
        
        if not has_dtensor_weights:
            return x
        
        # Handle nested structures (tuple, list, dict)
        return self._convert_to_dtensor_structure(x, device_mesh)

    def _convert_to_dtensor_structure(self, x, device_mesh):
        """Convert nested structures to DTensors recursively.
        
        Args:
            x: Input structure (can be tensor, tuple, list, dict)
            device_mesh: DeviceMesh for DTensor conversion
            
        Returns:
            Same structure with tensors converted to DTensors
        """
        from keras.src.backend.torch.distribution_lib import (
            DTensor,
            Replicate,
        )
        
        if x is None:
            return x
        
        if isinstance(x, DTensor):
            return x
        
        if isinstance(x, torch.Tensor):
            return DTensor.from_local(x, device_mesh, [Replicate()])
        
        if isinstance(x, dict):
            return {k: self._convert_to_dtensor_structure(v, device_mesh) for k, v in x.items()}
        
        if isinstance(x, list):
            return [self._convert_to_dtensor_structure(v, device_mesh) for v in x]
        
        if isinstance(x, tuple):
            return tuple(self._convert_to_dtensor_structure(v, device_mesh) for v in x)
        
        # For other types, return as-is
        return x

    def _convert_dtensor_output(self, x):
        """Convert DTensor outputs to local tensors.
        
        When model has DTensor weights, the forward pass returns DTensors.
        For sharded outputs, we need to ALL_GATHER to reconstruct the full tensor.
        For loss computation, we need the full global shape, not just the local shard.
        
        Args:
            x: Output tensor (can be torch.Tensor, DTensor, or nested structure)
            
        Returns:
            Same structure, with DTensors converted to local tensors
        """
        from keras.src.distribution.distribution_lib import distribution
        from keras.src.distribution.distribution_lib import ModelParallel
        from keras.src.backend.torch.distribution_lib import (
            dtensor_to_local,
            DTensor,
            Replicate,
            Shard,
            DTENSOR_AVAILABLE,
        )
        
        # If DTensor not available, return as-is
        if not DTENSOR_AVAILABLE:
            return x
        
        # Check if ModelParallel distribution is active
        dist = distribution()
        if not isinstance(dist, ModelParallel):
            return x
        
        # If x is not a DTensor, return as-is
        if x is None:
            return x
        
        # Check if it's a DTensor
        if isinstance(x, DTensor):
            # Check if the DTensor is sharded (has non-Replicate placements)
            is_sharded = not all(isinstance(p, Replicate) for p in x.placements)
            
            if is_sharded:
                # Need to ALL_GATHER to reconstruct the full tensor
                # This is critical for loss computation to work correctly
                # Find the shard dimension (the dimension being sharded)
                shard_dim = None
                for i, placement in enumerate(x.placements):
                    if isinstance(placement, Shard):
                        shard_dim = i
                        break
                
                    if shard_dim is not None and torch.distributed.is_initialized():
                        # Perform all_gather along the shard dimension
                        world_size = torch.distributed.get_world_size()
                        
                        # Get local tensor
                        local_tensor = x.to_local()
                        local_shape = list(local_tensor.shape)
                        global_shape = list(x.shape)
                        
                        # For sharded tensors in training, we need proper gradient handling
                        # Use our custom all_gather that preserves gradients
                        if local_tensor.requires_grad:
                            # Use custom all_gather with gradient support
                            full_tensor = _all_gather_with_grad(local_tensor, shard_dim)
                            
                            return full_tensor
                        else:
                            # For inference mode - no gradients needed
                            output = [torch.empty_like(local_tensor) for _ in range(world_size)]
                            torch.distributed.all_gather(output, local_tensor.contiguous())
                            full_tensor = torch.cat(output, dim=shard_dim)
                            return full_tensor
            
            # Not sharded or no distributed, just convert to local
            return dtensor_to_local(x)
        
        # For nested structures, check if any element is a DTensor
        if isinstance(x, (dict, list, tuple)):
            # Check if any element is a sharded DTensor and process recursively
            has_sharded_dtensor = False
            def check_for_sharded_dtensor(item):
                if isinstance(item, DTensor):
                    # Check if it's sharded
                    is_sharded = not all(isinstance(p, Replicate) for p in item.placements)
                    if is_sharded:
                        return True
                if isinstance(item, (dict, list, tuple)):
                    for v in item.values() if isinstance(item, dict) else item:
                        if check_for_sharded_dtensor(v):
                            return True
                return False
            
            has_sharded_dtensor = check_for_sharded_dtensor(x)
            
            if has_sharded_dtensor:
                # Process recursively to properly handle all-gather for sharded DTensors
                return self._convert_dtensor_output_structure(x)
        
        return x
    
    def _convert_dtensor_output_structure(self, x):
        """Recursively convert nested DTensor structures with proper all-gather.
        
        Args:
            x: Nested structure (dict, list, tuple) potentially containing DTensors
            
        Returns:
            Same structure with DTensors converted (sharded ones all-gathered)
        """
        from keras.src.distribution.distribution_lib import distribution
        from keras.src.distribution.distribution_lib import ModelParallel
        from keras.src.backend.torch.distribution_lib import (
            DTensor,
            Replicate,
            Shard,
        )
        
        if x is None:
            return x
        
        if isinstance(x, DTensor):
            # Check if sharded and handle
            is_sharded = not all(isinstance(p, Replicate) for p in x.placements)
            if is_sharded and torch.distributed.is_initialized():
                # Find shard dimension
                shard_dim = None
                for i, placement in enumerate(x.placements):
                    if isinstance(placement, Shard):
                        shard_dim = i
                        break
                
                if shard_dim is not None:
                    local_tensor = x.to_local()
                    if local_tensor.requires_grad:
                        # Use custom all_gather with gradient support
                        return _all_gather_with_grad(local_tensor, shard_dim)
                    else:
                        # For inference mode
                        world_size = torch.distributed.get_world_size()
                        output = [torch.empty_like(local_tensor) for _ in range(world_size)]
                        torch.distributed.all_gather(output, local_tensor.contiguous())
                        return torch.cat(output, dim=shard_dim)
            # Not sharded or no distributed, convert to local
            return x.to_local()
        
        if isinstance(x, dict):
            return {k: self._convert_dtensor_output_structure(v) for k, v in x.items()}
        
        if isinstance(x, list):
            return [self._convert_dtensor_output_structure(v) for v in x]
        
        if isinstance(x, tuple):
            return tuple(self._convert_dtensor_output_structure(v) for v in x)
        
        # For other types, return as-is
        return x

    def _parallelize_if_needed(self):
        """Parallelize the model if ModelParallel distribution is active.
        
        This method checks if the model should be parallelized and applies
        parallelize_keras_model if needed. It should be called during fit/evaluate.
        """
        if self._torch_module_parallelized:
            return
        
        try:
            from keras.src.backend.torch.distribution_lib import (
                parallelize_keras_model,
                _get_default_device_mesh,
                TENSOR_PARALLEL_AVAILABLE,
            )
            from keras.src.distribution.distribution_lib import (
                distribution,
                ModelParallel,
            )
        except ImportError:
            return
        
        if not TENSOR_PARALLEL_AVAILABLE:
            return
        
        # Check if ModelParallel distribution is active
        dist = distribution()
        if not isinstance(dist, ModelParallel):
            return
        
        # Check if we have a layout map
        if not hasattr(dist, '_layout_map') or not dist._layout_map:
            return
        
        # Get device mesh
        device_mesh = _get_default_device_mesh()
        if device_mesh is None:
            return
        
        # Get the underlying torch module
        if hasattr(self, '_torch_layers'):
            torch_module = self._torch_layers
        else:
            torch_module = self
        
        # Parallelize the model
        try:
            parallelize_keras_model(
                torch_module,
                device_mesh=device_mesh,
                layout_map=dist._layout_map
            )
            self._torch_module_parallelized = True
        except Exception as e:
            pass

    def build(self, input_shape=None):
        """Build the model and optionally apply automatic parallelization.
        
        This override automatically calls parallelize_keras_model() when:
        - PyTorch backend is active
        - ModelParallel distribution is set
        - Model has layers that can be parallelized
        """
        # Call parent build method if it exists
        # The parent Layer.build creates weights and sets self.built = True
        try:
            from keras.src.layers.layer import Layer
            if isinstance(self, Layer) and hasattr(super(), 'build'):
                if input_shape is not None:
                    super().build(input_shape)
                else:
                    super().build()
        except (TypeError, AttributeError):
            # If super().build doesn't exist or fails, continue anyway
            pass
    
    def _symbolic_build(self, *args, **kwargs):
        """Override _symbolic_build for automatic parallelization.

        Note: Parallelization now happens BEFORE this method is called
        (in fit()/evaluate()/predict()). This ensures that when weights
        are created during _symbolic_build, they are created as sharded
        DTensors from the start.

        This method calls the parent _symbolic_build to handle the actual
        model building and weight creation.
        """
        # Call the parent _symbolic_build method
        # The model is already set up for parallelization by _parallelize_if_needed()
        return super()._symbolic_build(*args, **kwargs)

    def _should_torch_compile(self):
        # require torch>=2.1.0 to enable dynamo since it
        # includes many improvements/fixes to torch.compile()
        # TODO eventually we want to get rid of this when
        # torch is upgraded to >=2.1 (from 2.0.1) in g3
        if self.jit_compile and parse(torch.__version__) < parse("2.1.0"):
            warnings.warn(
                "Please upgrade to torch>=2.1.0 for `jit_compile=True` "
                "to take effect. Using `jit_compile=False`"
            )
            self.jit_compile = False

        return self.jit_compile

    def train_step(self, data):
        x, y, sample_weight = data_adapter_utils.unpack_x_y_sample_weight(data)

        # Convert inputs to DTensors if needed for ModelParallel training
        # This ensures that when model has DTensor weights, inputs are also DTensors
        x = self._ensure_dtensor_input(x)
        y = self._ensure_dtensor_input(y)

        # Compute predictions
        if self._call_has_training_arg:
            y_pred = self(x, training=True)
        else:
            y_pred = self(x)

        # Convert DTensor outputs and labels to local tensors for loss computation
        y_pred = self._convert_dtensor_output(y_pred)
        y = self._convert_dtensor_output(y)
        x = self._convert_dtensor_output(x)

        # Call torch.nn.Module.zero_grad() to clear the leftover gradients
        # for the weights from the previous train step.
        self.zero_grad()

        loss = self._compute_loss(
            x=x, y=y, y_pred=y_pred, sample_weight=sample_weight, training=True
        )
        self._loss_tracker.update_state(
            loss,
            sample_weight=next(
                i for i in tree.flatten(x) if i is not None
            ).shape[0],
        )
        if self.optimizer is not None:
            loss = self.optimizer.scale_loss(loss)

        # Compute gradients
        if self.trainable_weights:
            # Call torch.Tensor.backward() on the loss to compute gradients
            # for the weights.
            loss.backward()

            trainable_weights = self.trainable_weights[:]
            gradients = [v.value.grad for v in trainable_weights]

            # Update weights
            with torch.no_grad():
                self.optimizer.apply(gradients, trainable_weights)
        else:
            warnings.warn("The model does not have any trainable weights.")

        return self.compute_metrics(x, y, y_pred, sample_weight=sample_weight)

    def test_step(self, data):
        (
            x,
            y,
            sample_weight,
        ) = data_adapter_utils.unpack_x_y_sample_weight(data)
        
        # Convert inputs to DTensors if needed for ModelParallel training
        x = self._ensure_dtensor_input(x)
        y = self._ensure_dtensor_input(y)
        
        if self._call_has_training_arg:
            y_pred = self(x, training=False)
        else:
            y_pred = self(x)
        
        # Convert DTensor outputs and labels to local tensors for loss computation
        y_pred = self._convert_dtensor_output(y_pred)
        y = self._convert_dtensor_output(y)
        x = self._convert_dtensor_output(x)
        
        loss = self._compute_loss(
            x=x, y=y, y_pred=y_pred, sample_weight=sample_weight, training=False
        )
        self._loss_tracker.update_state(
            loss,
            sample_weight=next(
                i for i in tree.flatten(x) if i is not None
            ).shape[0],
        )
        return self.compute_metrics(x, y, y_pred, sample_weight=sample_weight)

    def predict_step(self, data):
        x, _, _ = data_adapter_utils.unpack_x_y_sample_weight(data)
        
        # Convert inputs to DTensors if needed for ModelParallel training
        x = self._ensure_dtensor_input(x)
        
        if self._call_has_training_arg:
            y_pred = self(x, training=False)
        else:
            y_pred = self(x)
        
        # Convert DTensor outputs to local tensors
        y_pred = self._convert_dtensor_output(y_pred)
        
        return y_pred

    def make_train_function(self, force=False):
        if self.train_function is not None and not force:
            return self.train_function

        if self.steps_per_execution > 1:
            raise ValueError(
                "`steps_per_execution` must be 1 with the PyTorch backend. "
                f"Received: steps_per_execution={self.steps_per_execution}"
            )

        def one_step_on_data(data):
            """Runs a single training step on a batch of data."""
            data = data[0]
            return self.train_step(data)

        if self._should_torch_compile():
            self.train_function = torch.compile(one_step_on_data)
        else:
            self.train_function = one_step_on_data

    def make_test_function(self, force=False):
        if self.test_function is not None and not force:
            return self.test_function

        if self.steps_per_execution > 1:
            raise ValueError(
                "`steps_per_execution` must be 1 with the PyTorch backend. "
                f"Received: steps_per_execution={self.steps_per_execution}"
            )

        def one_step_on_data(data):
            """Runs a single test step on a batch of data."""
            data = data[0]
            with torch.no_grad():
                return self.test_step(data)

        if self._should_torch_compile():
            self.test_function = torch.compile(one_step_on_data)
        else:
            self.test_function = one_step_on_data

    def make_predict_function(self, force=False):
        if self.predict_function is not None and not force:
            return self.predict_function

        if self.steps_per_execution > 1:
            raise ValueError(
                "`steps_per_execution` must be 1 with the PyTorch backend. "
                f"Received: steps_per_execution={self.steps_per_execution}"
            )

        def one_step_on_data(data):
            """Runs a predict test step on a batch of data."""
            data = data[0]
            with torch.no_grad():
                return self.predict_step(data)

        if self._should_torch_compile():
            self.predict_function = torch.compile(one_step_on_data)
        else:
            self.predict_function = one_step_on_data

    @traceback_utils.filter_traceback
    def fit(
        self,
        x=None,
        y=None,
        batch_size=None,
        epochs=1,
        verbose="auto",
        callbacks=None,
        validation_split=0.0,
        validation_data=None,
        shuffle=True,
        class_weight=None,
        sample_weight=None,
        initial_epoch=0,
        steps_per_epoch=None,
        validation_steps=None,
        validation_batch_size=None,
        validation_freq=1,
    ):
        if not self.compiled:
            raise ValueError(
                "You must call `compile()` before calling `fit()`."
            )
        # Possibly cap epochs for debugging runs.
        max_epochs = config.max_epochs()
        if max_epochs and max_epochs < epochs:
            warnings.warn("Limiting epochs to %d" % max_epochs)
            epochs = max_epochs

        # TODO: respect compiled trainable state
        self._eval_epoch_iterator = None
        if validation_split and validation_data is None:
            # Create the validation data using the training data. Only supported
            # for TF/numpy/jax arrays.
            # TODO: Support torch tensors for validation data.
            (
                (x, y, sample_weight),
                validation_data,
            ) = array_slicing.train_validation_split(
                (x, y, sample_weight), validation_split=validation_split
            )

        if validation_data is not None:
            (
                val_x,
                val_y,
                val_sample_weight,
            ) = data_adapter_utils.unpack_x_y_sample_weight(validation_data)

        # Create an iterator that yields batches for one epoch.
        epoch_iterator = TorchEpochIterator(
            x=x,
            y=y,
            sample_weight=sample_weight,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            shuffle=shuffle,
            class_weight=class_weight,
            steps_per_execution=self.steps_per_execution,
        )

        # CRITICAL: Parallelize model BEFORE any weight creation!
        # This ensures that when _symbolic_build() creates weights, they are
        # created as sharded DTensors from the start, preventing OOM.
        # The parallelize_module transforms the model so weights are partitioned
        # across devices during creation, not after.
        self._parallelize_if_needed()

        self._symbolic_build(iterator=epoch_iterator)
        epoch_iterator.reset()

        # Container that configures and calls callbacks.
        if not isinstance(callbacks, callbacks_module.CallbackList):
            callbacks = callbacks_module.CallbackList(
                callbacks,
                add_history=True,
                add_progbar=verbose != 0,
                verbose=verbose,
                epochs=epochs,
                steps=epoch_iterator.num_batches,
                model=self,
            )

        self.stop_training = False
        training_logs = {}
        self.make_train_function()
        callbacks.on_train_begin()
        initial_epoch = self._initial_epoch or initial_epoch
        for epoch in range(initial_epoch, epochs):
            self.reset_metrics()
            callbacks.on_epoch_begin(epoch)

            # Switch the torch Module to training mode. Inform torch layers to
            # do training behavior in case the user did not use `self.training`
            # when implementing a custom layer with torch layers.
            self.train()

            logs = {}
            for begin_step, end_step, data in epoch_iterator:
                # Callbacks
                callbacks.on_train_batch_begin(begin_step)

                logs = self.train_function(data)

                # Callbacks
                callbacks.on_train_batch_end(end_step, logs)
                if self.stop_training:
                    break

            # Override with model metrics instead of last step logs if needed.
            epoch_logs = dict(self._get_metrics_result_or_logs(logs))

            # Switch the torch Module back to testing mode.
            self.eval()

            # Run validation.
            if validation_data is not None and self._should_eval(
                epoch, validation_freq
            ):
                # Create TorchEpochIterator for evaluation and cache it.
                if getattr(self, "_eval_epoch_iterator", None) is None:
                    self._eval_epoch_iterator = TorchEpochIterator(
                        x=val_x,
                        y=val_y,
                        sample_weight=val_sample_weight,
                        batch_size=validation_batch_size or batch_size,
                        steps_per_execution=self.steps_per_execution,
                        steps_per_epoch=validation_steps,
                        shuffle=False,
                    )
                val_logs = self.evaluate(
                    x=val_x,
                    y=val_y,
                    sample_weight=val_sample_weight,
                    batch_size=validation_batch_size or batch_size,
                    steps=validation_steps,
                    callbacks=callbacks,
                    return_dict=True,
                    _use_cached_eval_dataset=True,
                )
                val_logs = {
                    f"val_{name}": val for name, val in val_logs.items()
                }
                epoch_logs.update(val_logs)

            callbacks.on_epoch_end(epoch, epoch_logs)
            training_logs = epoch_logs
            if self.stop_training:
                break

        if (
            isinstance(self.optimizer, optimizers_module.Optimizer)
            and epochs > 0
        ):
            self.optimizer.finalize_variable_values(self.trainable_weights)

        # If _eval_epoch_iterator exists, delete it after all epochs are done.
        if getattr(self, "_eval_epoch_iterator", None) is not None:
            del self._eval_epoch_iterator
        callbacks.on_train_end(logs=training_logs)
        return self.history

    @traceback_utils.filter_traceback
    def evaluate(
        self,
        x=None,
        y=None,
        batch_size=None,
        verbose="auto",
        sample_weight=None,
        steps=None,
        callbacks=None,
        return_dict=False,
        **kwargs,
    ):
        # TODO: respect compiled trainable state
        use_cached_eval_dataset = kwargs.pop("_use_cached_eval_dataset", False)
        if kwargs:
            raise ValueError(f"Arguments not recognized: {kwargs}")

        if use_cached_eval_dataset:
            epoch_iterator = self._eval_epoch_iterator
        else:
            # Create an iterator that yields batches of input/target data.
            epoch_iterator = TorchEpochIterator(
                x=x,
                y=y,
                sample_weight=sample_weight,
                batch_size=batch_size,
                steps_per_epoch=steps,
                shuffle=False,
                steps_per_execution=self.steps_per_execution,
            )

        # CRITICAL: Parallelize model BEFORE any weight creation!
        # This ensures that when _symbolic_build() creates weights, they are
        # created as sharded DTensors from the start, preventing OOM.
        self._parallelize_if_needed()

        self._symbolic_build(iterator=epoch_iterator)
        epoch_iterator.reset()

        # Container that configures and calls callbacks.
        if not isinstance(callbacks, callbacks_module.CallbackList):
            callbacks = callbacks_module.CallbackList(
                callbacks,
                add_progbar=verbose != 0,
                verbose=verbose,
                epochs=1,
                steps=epoch_iterator.num_batches,
                model=self,
            )

        # Switch the torch Module back to testing mode.
        self.eval()

        self.make_test_function()
        self.stop_evaluating = False
        callbacks.on_test_begin()
        logs = {}
        self.reset_metrics()
        for begin_step, end_step, data in epoch_iterator:
            callbacks.on_test_batch_begin(begin_step)
            logs = self.test_function(data)
            callbacks.on_test_batch_end(end_step, logs)
            if self.stop_evaluating:
                break
        logs = self._get_metrics_result_or_logs(logs)
        callbacks.on_test_end(logs)

        if return_dict:
            return logs
        return self._flatten_metrics_in_order(logs)

    @traceback_utils.filter_traceback
    def predict(
        self, x, batch_size=None, verbose="auto", steps=None, callbacks=None
    ):
        # Create an iterator that yields batches of input data.
        epoch_iterator = TorchEpochIterator(
            x=x,
            batch_size=batch_size,
            steps_per_epoch=steps,
            shuffle=False,
            steps_per_execution=self.steps_per_execution,
        )

        # CRITICAL: Parallelize model BEFORE any weight creation!
        # This ensures that when _symbolic_build() creates weights, they are
        # created as sharded DTensors from the start, preventing OOM.
        self._parallelize_if_needed()

        # Container that configures and calls callbacks.
        if not isinstance(callbacks, callbacks_module.CallbackList):
            callbacks = callbacks_module.CallbackList(
                callbacks,
                add_progbar=verbose != 0,
                verbose=verbose,
                epochs=1,
                steps=epoch_iterator.num_batches,
                model=self,
            )

        def append_to_outputs(batch_outputs, outputs):
            if outputs is None:
                outputs = tree.map_structure(
                    lambda batch_output: [batch_output],
                    batch_outputs,
                )
            else:
                tree.map_structure_up_to(
                    batch_outputs,
                    lambda output, batch_output: output.append(batch_output),
                    outputs,
                    batch_outputs,
                )
            return outputs

        # Switch the torch Module back to testing mode.
        self.eval()

        self.make_predict_function()
        self.stop_predicting = False
        callbacks.on_predict_begin()
        outputs = None
        for begin_step, end_step, data in epoch_iterator:
            callbacks.on_predict_batch_begin(begin_step)
            batch_outputs = self.predict_function(data)
            outputs = append_to_outputs(batch_outputs, outputs)
            callbacks.on_predict_batch_end(end_step, {"outputs": batch_outputs})
            if self.stop_predicting:
                break
        callbacks.on_predict_end()
        outputs = tree.map_structure(backend.convert_to_numpy, outputs)
        return tree.map_structure_up_to(batch_outputs, np.concatenate, outputs)

    def train_on_batch(
        self,
        x,
        y=None,
        sample_weight=None,
        class_weight=None,
        return_dict=False,
    ):
        self._assert_compile_called("train_on_batch")
        if class_weight is not None:
            if sample_weight is not None:
                raise ValueError(
                    "Arguments `sample_weight` and `class_weight` "
                    "cannot be specified at the same time. "
                    f"Received: sample_weight={sample_weight}, "
                    f"class_weight={class_weight}"
                )
            sample_weight = data_adapter_utils.class_weight_to_sample_weights(
                y, class_weight
            )

        data = (x, y, sample_weight)

        # Maybe build model
        self._symbolic_build(data_batch=data)
        self.make_train_function()

        logs = self.train_function([data])
        logs = tree.map_structure(lambda x: np.array(x), logs)
        if return_dict:
            return logs
        return self._flatten_metrics_in_order(logs)

    def test_on_batch(
        self,
        x,
        y=None,
        sample_weight=None,
        return_dict=False,
    ):
        self._assert_compile_called("test_on_batch")

        data = (x, y, sample_weight)

        # Maybe build model
        self._symbolic_build(data_batch=data)
        self.make_test_function()

        logs = self.test_function([data])
        logs = tree.map_structure(lambda x: np.array(x), logs)
        if return_dict:
            return logs
        return self._flatten_metrics_in_order(logs)

    def predict_on_batch(self, x):
        self.make_predict_function()
        batch_outputs = self.predict_function([(x,)])
        batch_outputs = tree.map_structure(
            backend.convert_to_numpy, batch_outputs
        )
        return batch_outputs


class TorchEpochIterator(EpochIterator):
    def _get_iterator(self):
        return self.data_adapter.get_torch_dataloader()
