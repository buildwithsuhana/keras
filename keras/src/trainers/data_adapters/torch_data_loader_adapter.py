import itertools

import numpy as np

from keras.src import tree
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.trainers.data_adapters.data_adapter import DataAdapter


class TorchDataLoaderAdapter(DataAdapter):
    """Adapter that handles `torch.utils.data.DataLoader`."""

    def __init__(self, dataloader):
        import torch

        if not isinstance(dataloader, torch.utils.data.DataLoader):
            raise ValueError(
                f"Expected argument `dataloader` to be an instance of"
                f"`torch.utils.data.DataLoader`. Received: {dataloader}"
            )

        self._dataloader = dataloader
        self._output_signature = None
        self._batch_size = dataloader.batch_size
        self._num_batches = None
        self._partial_batch_size = None
        if hasattr(dataloader.dataset, "__len__"):
            self._num_batches = len(dataloader)
            if self._batch_size is not None:
                self._partial_batch_size = (
                    len(dataloader.dataset) % self._batch_size
                )

    def get_numpy_iterator(self):
        """Get a numpy iterator over the DataLoader batches.

        When DTensor distribution is active, this uses the DTensor-aware
        wrapper to properly handle mixed tensor types during iteration.
        """
        # Use get_torch_dataloader() which returns the DTensor-aware wrapper
        # when distribution is active, preventing "aten.index.Tensor: got
        # mixed torch.Tensor and DTensor" errors.
        for batch in self.get_torch_dataloader():
            # shared memory using `np.asarray`
            yield tuple(
                tree.map_structure(
                    lambda x: np.asarray(x.cpu()), batch, none_is_leaf=False
                )
            )

    def get_jax_iterator(self):
        # We use numpy as an intermediary because it is faster.
        return self.get_numpy_iterator()

    def get_tf_dataset(self):
        from keras.src.utils.module_utils import tensorflow as tf

        if self._output_signature is None:
            batches = list(
                itertools.islice(
                    self._dataloader,
                    data_adapter_utils.NUM_BATCHES_FOR_TENSOR_SPEC,
                )
            )
            self._output_signature = tuple(
                data_adapter_utils.get_tensor_spec(batches)
            )
        return tf.data.Dataset.from_generator(
            self.get_numpy_iterator,
            output_signature=self._output_signature,
        )

    def get_torch_dataloader(self):
        """Return the underlying DataLoader.

        For DTensor distribution, we need to handle the case where the
        DataLoader's internal __getitems__ uses tree_map which fails with
        mixed DTensor and regular torch.Tensor.
        """
        from keras.src.distribution import distribution_lib
        from keras.src.backend.torch import distribution_lib as backend_dist

        dist = distribution_lib.distribution()
        if dist is None:
            return self._dataloader

        # Check if we're in model parallel mode with DTensor
        try:
            from torch.distributed._tensor import DTensor
        except ImportError:
            return self._dataloader

        # Get the data layout and check if we need DTensor conversion
        x_layout = dist.get_data_layout((None,))
        if x_layout is None:
            return self._dataloader

        mesh, placements = backend_dist._get_dtensor_mesh_and_placements(
            x_layout
        )
        if mesh is None:
            return self._dataloader

        # Wrap the dataloader with DTensor conversion
        return _DTensorAwareDataLoader(self._dataloader, mesh, placements)

    @property
    def builtin_prefetch(self):
        prefetch_factor = self._dataloader.prefetch_factor
        if prefetch_factor is not None and prefetch_factor > 0:
            return True
        else:
            return False

    @property
    def num_batches(self):
        return self._num_batches

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def has_partial_batch(self):
        if self._partial_batch_size:
            return self._partial_batch_size > 0
        else:
            return None

    @property
    def partial_batch_size(self):
        return self._partial_batch_size


class _DTensorAwareDataLoader:
    """A wrapper around torch DataLoader that converts data to DTensor.

    This wrapper is needed because PyTorch's DataLoader internally uses
    tree_map in __getitems__ which fails with mixed DTensor and regular
    torch.Tensor. By converting the data to DTensor after collection,
    we avoid this issue.
    """

    def __init__(self, dataloader, mesh, placements):
        import torch

        self._dataloader = dataloader
        self._mesh = mesh
        self._placements = placements

        # Copy all attributes from the original dataloader
        self.batch_size = dataloader.batch_size
        self.num_batches = len(dataloader)
        self.dataset = _DTensorAwareDataset(dataloader.dataset, mesh, placements)
        self.drop_last = dataloader.drop_last
        self.prefetch_factor = dataloader.prefetch_factor
        self._iterator = None

    def __iter__(self):
        # Use self.dataset (the _DTensorAwareDataset wrapper) instead of
        # self._dataloader to ensure that __getitems__ calls go through
        # our DTensor-aware wrapper. This prevents the "aten.index.Tensor:
        # got mixed torch.Tensor and DTensor" error that occurs when
        # PyTorch's internal tree_map in DataLoader.__next__ mixes types.
        self._iterator = iter(self.dataset)
        return self

    def __next__(self):
        batch = next(self._iterator)
        # Convert the batch to DTensor after it's collected
        from keras.src.backend.torch import distribution_lib as backend_dist
        return backend_dist._convert_batch_to_dtensor(
            batch, self._mesh, self._placements
        )

    def __len__(self):
        return len(self._dataloader)


class _DTensorAwareDataset:
    """A wrapper around torch Dataset that converts data to DTensor.

    This wrapper handles the __getitems__ method to ensure data is
    properly converted to DTensor when needed. It also handles the case
    where indices may be DTensors by converting them back to regular
    torch.Tensors before passing to the underlying dataset.
    """

    def __init__(self, dataset, mesh, placements):
        self._dataset = dataset
        self._mesh = mesh
        self._placements = placements

        # Copy relevant attributes
        if hasattr(dataset, "__len__"):
            self.__len__ = lambda: len(dataset)

    def _convert_indices_to_torch(self, indices):
        """Convert DTensor indices back to regular torch.Tensor.

        This is necessary because PyTorch's tree_map in DataLoader
        may pass DTensor indices, but the underlying dataset expects
        regular torch.Tensor indices.
        """
        from keras.src.backend.torch import distribution_lib as backend_dist

        def convert_index(idx):
            if backend_dist._is_dtensor(idx):
                # Get the local tensor from DTensor
                return idx.to_local()
            return idx

        from keras.src import tree
        return tree.map_structure(convert_index, indices, none_is_leaf=False)

    def __getitem__(self, index):
        # Convert index if it's a DTensor
        index = self._convert_indices_to_torch(index)
        item = self._dataset[index]
        from keras.src.backend.torch import distribution_lib as backend_dist
        return backend_dist._convert_batch_to_dtensor(
            item, self._mesh, self._placements
        )

    def __getitems__(self, indices):
        """Get multiple items at once, converting to DTensor.

        This is the critical method that was causing the mixed tensor error.
        We override it to ensure proper DTensor conversion.

        We also need to convert indices from DTensor to regular torch.Tensor
        before passing them to the underlying dataset's __getitems__, since
        PyTorch's tree_map may pass DTensor indices.
        """
        # Convert indices from DTensor to regular torch.Tensor
        indices = self._convert_indices_to_torch(indices)

        items = self._dataset.__getitems__(indices)
        from keras.src.backend.torch import distribution_lib as backend_dist
        return backend_dist._convert_batch_to_dtensor(
            items, self._mesh, self._placements
        )


