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

        For distributed training, this returns the native sharded data
        without DTensor conversion for better compatibility.
        """
        for batch in self.get_torch_dataloader():
            # Convert to numpy for compatibility
            yield tuple(
                tree.map_structure(
                    lambda x: np.asarray(x.cpu()) if hasattr(x, "cpu") else np.asarray(x), 
                    batch, 
                    none_is_leaf=False
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
        
        For distributed training with native sharding, this returns
        a wrapper that properly distributes data across processes.
        """
        from keras.src.distribution import distribution_lib
        from keras.src.backend.torch import distribution_lib as backend_dist

        dist = distribution_lib.distribution()
        if dist is None:
            return self._dataloader

        # Check distribution type and get data layout
        data_layout = dist.get_data_layout((None,))
        if data_layout is None:
            return self._dataloader

        # Check if DTensor is available and should be used
        use_dtensor = getattr(dist, '_use_dtensor', False)
        
        if use_dtensor and backend_dist.is_dtensor_available():
            # Use DTensor-based distribution if explicitly requested
            mesh, placements = backend_dist._get_dtensor_mesh_and_placements(
                data_layout
            )
            if mesh is not None:
                return _DTensorAwareDataLoader(self._dataloader, mesh, placements)
        
        # Use native PyTorch sharding (default)
        return _ShardAwareDataLoader(self._dataloader, dist)

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


class _ShardAwareDataLoader:
    """A wrapper around torch DataLoader that handles native sharding.
    
    This wrapper distributes data across processes using native PyTorch
    operations (torch.split/torch.chunk) instead of DTensor. This provides
    better compatibility with torch.compile and avoids "mixed tensor" errors.
    """

    def __init__(self, dataloader, distribution):
        import torch
        import torch.distributed as dist

        self._dataloader = dataloader
        self._distribution = distribution
        
        # Copy all attributes from the original dataloader
        self.batch_size = dataloader.batch_size
        self.num_batches = len(dataloader)
        self.drop_last = dataloader.drop_last
        self.prefetch_factor = dataloader.prefetch_factor
        self._iterator = None
        
        # Get distribution info
        self._rank = dist.get_rank() if dist.is_initialized() else 0
        self._world_size = dist.get_world_size() if dist.is_initialized() else 1
        self._local_rank = self._rank % torch.cuda.device_count() if torch.cuda.is_available() else 0
        
        # Get sharding info from distribution
        self._batch_dim_name = distribution.batch_dim_name
        self._device_mesh = distribution.device_mesh
        
        # Wrap the dataset for sharding
        self.dataset = _ShardAwareDataset(
            dataloader.dataset, 
            distribution, 
            self._rank, 
            self._world_size
        )

    def __iter__(self):
        # Use our sharding-aware dataset wrapper
        self._iterator = iter(self.dataset)
        return self

    def __next__(self):
        batch = next(self._iterator)
        return batch

    def __len__(self):
        return len(self._dataloader)


class _ShardAwareDataset:
    """A wrapper around torch Dataset that handles native sharding.
    
    This wrapper distributes data across processes using native PyTorch
    operations. For data parallelism, each process gets a shard of the data.
    """

    def __init__(self, dataset, distribution, rank, world_size):
        self._dataset = dataset
        self._distribution = distribution
        self._rank = rank
        self._world_size = world_size
        
        # Get batch dimension info
        self._batch_dim_name = distribution.batch_dim_name
        self._device_mesh = distribution.device_mesh
        
        # Find batch dimension index in mesh
        if self._batch_dim_name and self._batch_dim_name in self._device_mesh.axis_names:
            self._batch_dim_idx = list(self._device_mesh.axis_names).index(self._batch_dim_name)
        else:
            self._batch_dim_idx = 0
        
        # Calculate shard info
        if hasattr(dataset, "__len__"):
            total_size = len(dataset)
            # Each process gets a shard of the dataset
            shard_size = total_size // world_size
            self._start_idx = shard_size * rank
            self._end_idx = shard_size * (rank + 1) if rank < world_size - 1 else total_size
            
            # Update __len__ to return local dataset size
            self.__len__ = lambda: max(0, self._end_idx - self._start_idx)
            
            if rank == 0:
                print(f"[ShardAwareDataset] Total dataset size: {total_size}, "
                      f"World size: {world_size}, Local size: {self.__len__()}")

    def __getitem__(self, index):
        """Get item from the local shard of the dataset."""
        # Adjust index for sharding
        local_idx = index + self._start_idx
        return self._dataset[local_idx]

    def __getitems__(self, indices):
        """Get multiple items from the local shard.
        
        This is called by PyTorch's DataLoader for batch collation.
        """
        # Adjust indices for sharding
        local_indices = [i + self._start_idx for i in indices]
        return self._dataset.__getitems__(local_indices)


class _DTensorAwareDataLoader:
    """A wrapper around torch DataLoader that converts data to DTensor.
    
    This is kept for backward compatibility when DTensor-based distribution
    is explicitly requested. For normal use, prefer _ShardAwareDataLoader.
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
        # our DTensor-aware wrapper.
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
    
    This is kept for backward compatibility. Converts data to DTensor
    after retrieval to avoid PyTorch's tree_map issues.
    """

    def __init__(self, dataset, mesh, placements):
        self._dataset = dataset
        self._mesh = mesh
        self._placements = placements

        # Copy relevant attributes
        if hasattr(dataset, "__len__"):
            self.__len__ = lambda: len(dataset)

    def __getitem__(self, index):
        # Get item from underlying dataset first (returns regular tensors)
        item = self._dataset[index]
        # Then convert to DTensor after retrieval
        from keras.src.backend.torch import distribution_lib as backend_dist
        return backend_dist._convert_batch_to_dtensor(
            item, self._mesh, self._placements
        )

    def __getitems__(self, indices):
        """Get multiple items at once, converting to DTensor."""
        # Get items from underlying dataset (returns regular tensors)
        items = self._dataset.__getitems__(indices)
        # Convert entire result to DTensors after retrieval
        from keras.src.backend.torch import distribution_lib as backend_dist
        return backend_dist._convert_batch_to_dtensor(
            items, self._mesh, self._placements
        )

