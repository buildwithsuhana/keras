import warnings

import numpy as np
import torch
from packaging.version import parse

from keras.src import backend
from keras.src import callbacks as callbacks_module
from keras.src import optimizers as optimizers_module
from keras.src import tree
from keras.src.backend import config
from keras.src.backend.torch import distribution_lib
from keras.src.trainers import trainer as base_trainer
from keras.src.trainers.data_adapters import array_slicing
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.trainers.epoch_iterator import EpochIterator
from keras.src.utils import traceback_utils


class TorchTrainer(base_trainer.Trainer):
    def __init__(self):
        super().__init__()
        self.train_function = None
        self.test_function = None
        self.predict_function = None
        self._torch_module_parallelized = False
        # Cache for torch.compile decision - set during fit/evaluate/predict
        # when distribution scope is still active
        self._torch_compile_disabled_for_mp = False
        # Cache for ModelParallel multi-process mode - used to skip input DTensor
        # conversion during training when distribution scope isn't available
        self._is_mp_multi_process = False

    def _parallelize_if_needed(self):
        """Parallelize the model if ModelParallel distribution is active.
        
        This method checks if the model should be parallelized and applies
        parallelize_keras_model if needed. It should be called during fit/evaluate.
        """
        if self._torch_module_parallelized:
            return
        from keras.src.backend.torch.distribution_lib import parallelize_torch_module, _get_default_device_mesh, TENSOR_PARALLEL_AVAILABLE
        from keras.src.distribution.distribution_lib import distribution,ModelParallel
        if not TENSOR_PARALLEL_AVAILABLE:
            return
        dist = distribution()
        if not isinstance(dist, ModelParallel):
            return
        if not hasattr(dist, '_layout_map') or not dist._layout_map:
            return
        device_mesh = _get_default_device_mesh()
        if device_mesh is None:
            return  
        if hasattr(self, '_torch_layers'):
            torch_module = self._torch_layers
        else:
            torch_module = self
        
        parallelize_torch_module(
                torch_module,
                device_mesh=device_mesh,
                layout_map=dist._layout_map
            )
        self._torch_module_parallelized = True

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

        # CRITICAL FIX: Disable torch.compile for ModelParallel in multi-process mode.
        # torch.compile uses fake tensors for tracing which doesn't work well with
        # DTensor operations when weights are sharded across multiple processes.
        # The sharding propagator fails during tracing because it can't handle
        # the shape mismatch between replicated inputs and sharded weights.
        if self.jit_compile:
            import torch.distributed as dist
            
            # Check if distributed is initialized (multi-process mode)
            if dist.is_available() and dist.is_initialized():
                # In multi-process mode, check if there's a cached ModelParallel mesh
                # from a previous distribution scope. We can detect this by checking
                # if the global state has a torch mesh with 1D shape (which is what
                # ModelParallel uses in multi-process mode).
                from keras.src.backend.common import global_state
                cached_mesh = global_state.get_global_attribute("torch_device_mesh", None)
                if cached_mesh is not None and hasattr(cached_mesh, 'mesh'):
                    # Check if it's a 1D mesh (ModelParallel in multi-process mode)
                    if cached_mesh.mesh.ndim == 1:
                        # This is ModelParallel in multi-process mode - disable torch.compile
                        warnings.warn(
                            "Disabling torch.compile for ModelParallel in multi-process mode. "
                            "torch.compile does not support DTensor operations with sharded weights "
                            "in multi-process training."
                        )
                        return False

        return self.jit_compile

    def _check_and_disable_torch_compile_for_mp(self):
        """Check if torch.compile should be disabled for ModelParallel in multi-process.
        
        This method should be called at the start of fit/evaluate/predict when
        the distribution scope is still active. It caches the decision so that
        _should_torch_compile() can use it later when the scope has exited.
        """
        if self._torch_compile_disabled_for_mp:
            # Already checked and disabled
            return
            
        if not self.jit_compile:
            # torch.compile not enabled, nothing to do
            return
            
        from keras.src.distribution.distribution_lib import distribution, ModelParallel
        import torch.distributed as dist
        
        current_dist = distribution()
        is_mp = isinstance(current_dist, ModelParallel)
        is_distributed = dist.is_available() and dist.is_initialized()
        
        if is_mp and is_distributed:
            warnings.warn(
                "Disabling torch.compile for ModelParallel in multi-process mode. "
                "torch.compile does not support DTensor operations with sharded weights "
                "in multi-process training."
            )
            self._torch_compile_disabled_for_mp = True

    def _cache_mp_multi_process_state(self):
        """Cache whether we're in ModelParallel multi-process mode.
        
        This method should be called at the start of fit/evaluate/predict when
        the distribution scope is still active. It caches the state so that
        prepare_input_for_distribution can use it later when the scope has exited
        (e.g., during torch.compile traced execution).
        """
        if self._is_mp_multi_process:
            # Already cached
            return
            
        from keras.src.distribution.distribution_lib import distribution, ModelParallel
        import torch.distributed as dist
        
        current_dist = distribution()
        is_mp = isinstance(current_dist, ModelParallel)
        is_distributed = dist.is_available() and dist.is_initialized()
        
        # Cache the state for later use
        self._is_mp_multi_process = is_mp and is_distributed
        
        # Also set the global state so prepare_input_for_distribution can access it
        from keras.src.backend.torch import distribution_lib as torch_dist_lib
        torch_dist_lib.set_mp_multi_process_state(self._is_mp_multi_process)
        
        if self._is_mp_multi_process:
            import os
            if os.environ.get("KERAS_DISTRIBUTION_DEBUG", "0") == "1":
                print("DEBUG | Cached _is_mp_multi_process=True for later use in prepare_input_for_distribution")

    def train_step(self, data):
        x, y, sample_weight = data_adapter_utils.unpack_x_y_sample_weight(data)
        x = distribution_lib.prepare_input_for_distribution(x)
        y = distribution_lib.prepare_input_for_distribution(y)
        # Compute predictions
        if self._call_has_training_arg:
            y_pred = self(x, training=True)
        else:
            y_pred = self(x)

        y_pred = distribution_lib.prepare_output_for_loss(y_pred)
        y = distribution_lib.prepare_output_for_loss(y)
        x = distribution_lib.prepare_output_for_loss(x)
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

            # Note: Gradient conversion to DTensor is now handled inside the
            # optimizer's _parallel_update_step to properly check optimizer
            # state variables (which may be DTensors in DataParallel) rather
            # than model weights (which are not DTensors in DataParallel).

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
        x = distribution_lib.prepare_input_for_distribution(x)
        y = distribution_lib.prepare_input_for_distribution(y)
        if self._call_has_training_arg:
            y_pred = self(x, training=False)
        else:
            y_pred = self(x)
        y_pred = distribution_lib.prepare_output_for_loss(y_pred)
        y = distribution_lib.prepare_output_for_loss(y)
        x = distribution_lib.prepare_output_for_loss(x)
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
        x = distribution_lib.prepare_input_for_distribution(x)
        if self._call_has_training_arg:
            y_pred = self(x, training=False)
        else:
            y_pred = self(x)
        y_pred = distribution_lib.prepare_output_for_loss(y_pred)
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

        self._parallelize_if_needed()
        # Check if torch.compile should be disabled for ModelParallel in multi-process
        # This must be done while the distribution scope is still active
        self._check_and_disable_torch_compile_for_mp()
        # Cache MP multi-process state while distribution scope is still active
        self._cache_mp_multi_process_state()
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

        self._parallelize_if_needed()
        # Check if torch.compile should be disabled for ModelParallel in multi-process
        # This must be done while the distribution scope is still active
        self._check_and_disable_torch_compile_for_mp()
        # Cache MP multi-process state while distribution scope is still active
        self._cache_mp_multi_process_state()
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

        self._parallelize_if_needed()
        # Check if torch.compile should be disabled for ModelParallel in multi-process
        # This must be done while the distribution scope is still active
        self._check_and_disable_torch_compile_for_mp()
        # Cache MP multi-process state while distribution scope is still active
        self._cache_mp_multi_process_state()
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
