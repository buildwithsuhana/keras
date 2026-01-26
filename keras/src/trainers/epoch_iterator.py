"""
Separation of concerns:

DataAdapter:
    - x, y
    - sample_weight
    - class_weight
    - shuffle
    - batch_size
        - steps, as it relates to batch_size for array data

EpochIterator:
    - whether to yield numpy or tf data
    - steps
    - most argument validation

Trainer:
    - steps_per_execution
    - validation_split
    - validation_data
    - validation_freq
    - epochs
    - initial_epoch
    - any backend-specific concern such as distribution

PyDataset:
    - num_workers
    - use_multiprocessing
    - max_queue_size

EpochIterator steps:

1. Look at data type and select correct DataHandler
2. Instantiate DataHandler with correct arguments
3. Raise or warn on unused arguments
4. in __iter__, iterate, either for a fixed number of steps
or until there is no data

"""

import contextlib
import warnings

import numpy as np

from keras.src import backend
from keras.src.backend import config
from keras.src.distribution import distribution_lib
from keras.src.trainers import data_adapters


class EpochIterator:
    def __init__(
        self,
        x,
        y=None,
        sample_weight=None,
        batch_size=None,
        steps_per_epoch=None,
        shuffle=False,
        class_weight=None,
        steps_per_execution=1,
    ):
        # Possibly cap steps_per_epoch for debugging runs.
        max_steps_per_epoch = config.max_steps_per_epoch()
        if max_steps_per_epoch:
            if not steps_per_epoch or max_steps_per_epoch < steps_per_epoch:
                warnings.warn(
                    "Limiting steps_per_epoch to %d" % max_steps_per_epoch
                )
                steps_per_epoch = max_steps_per_epoch
        self.steps_per_epoch = steps_per_epoch
        self.steps_per_execution = steps_per_execution
        self._current_iterator = None
        self._epoch_iterator = None
        self._steps_seen = 0
        self.data_adapter = data_adapters.get_data_adapter(
            x=x,
            y=y,
            sample_weight=sample_weight,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            shuffle=shuffle,
            class_weight=class_weight,
        )
        self._num_batches = self.data_adapter.num_batches

    def _get_iterator(self):
        return self.data_adapter.get_numpy_iterator()

    def _interrupted_warning(self):
        warnings.warn(
            "Your input ran out of data; interrupting training. "
            "Make sure that your dataset or generator can generate "
            "at least `steps_per_epoch * epochs` batches. "
            "You may need to use the `.repeat()` "
            "function when building your dataset.",
            stacklevel=2,
        )

    def reset(self):
        self._current_iterator = None
        self._num_batches = self.data_adapter.num_batches
        self._steps_seen = 0
        self._epoch_iterator = None
        self.data_adapter.on_epoch_end()

    def _distribute_data(self, data):
        """Distribute data batch across devices if Distribution is active.
        
        This function uses native PyTorch sharding by default for better
        compatibility with torch.compile. DTensor is only used if explicitly
        requested via distribution settings.
        """
        from keras.src import tree
        from keras.src.backend.torch import distribution_lib as backend_dist

        dist = distribution_lib.distribution()
        if dist is None:
            return data
        
        # Get the data layout
        data_layout = dist.get_data_layout(data.shape if hasattr(data, 'shape') else None)
        if data_layout is None:
            return data
        
        # Check if DTensor should be used (opt-in via distribution settings)
        use_dtensor = getattr(dist, '_use_dtensor', False)
        
        # Convert numpy arrays to torch tensors if needed
        def convert_to_tensor_if_needed(x):
            if hasattr(x, '__array__') or isinstance(x, np.ndarray):
                import torch
                return torch.from_numpy(np.asarray(x))
            return x

        data = tree.map_structure(
            convert_to_tensor_if_needed, data, none_is_leaf=False
        )

        # Use native PyTorch sharding by default
        return backend.distribution.distribute_data_input(
            data, data_layout, dist.batch_dim_name, use_dtensor=use_dtensor
        )

    def _enumerate_iterator(self):
        self.data_adapter.on_epoch_begin()
        steps_per_epoch = self.steps_per_epoch or self._num_batches or -1

        if steps_per_epoch > 0:
            if self._current_iterator is None or self.steps_per_epoch is None:
                self._current_iterator = iter(self._get_iterator())
                self._steps_seen = 0
            for step in range(0, steps_per_epoch, self.steps_per_execution):
                if self._num_batches and self._steps_seen >= self._num_batches:
                    if self.steps_per_epoch:
                        self._interrupted_warning()
                    break
                self._steps_seen += self.steps_per_execution
                yield (
                    step,
                    step + self.steps_per_execution - 1,
                    self._current_iterator,
                )
            if self._num_batches and self._steps_seen >= self._num_batches:
                self._current_iterator = iter(self._get_iterator())
                self._steps_seen = 0
        else:
            iterator = iter(self._get_iterator())
            step = -self.steps_per_execution
            while True:
                step += self.steps_per_execution
                self._steps_seen = step + self.steps_per_execution
                yield step, step + self.steps_per_execution - 1, iterator
        self.data_adapter.on_epoch_end()

    def __iter__(self):
        self._epoch_iterator = self._enumerate_iterator()
        return self

    def __next__(self):
        buffer = []
        begin_step, end_step, iterator = next(self._epoch_iterator)
        with self.catch_stop_iteration():
            for _ in range(self.steps_per_execution):
                data = next(iterator)
                data = self._distribute_data(data)
                buffer.append(data)
            return begin_step, end_step, buffer
        if buffer:
            return begin_step, end_step, buffer
        raise StopIteration

    def enumerate_epoch(self):
        for begin_step, end_step, data in self:
            yield begin_step, end_step, data

    @contextlib.contextmanager
    def catch_stop_iteration(self):
        """Catches errors when an iterator runs out of data."""
        try:
            yield
        except StopIteration:
            if self._num_batches is None:
                self._num_batches = self._steps_seen
            self._interrupted_warning()
            self._current_iterator = None
            self.data_adapter.on_epoch_end()

    @property
    def num_batches(self):
        if self.steps_per_epoch:
            return self.steps_per_epoch
        # Either copied from the data_adapter, or
        # inferred at the end of an iteration.
        return self._num_batches

