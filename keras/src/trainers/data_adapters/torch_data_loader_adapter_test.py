import math
from unittest.mock import patch

import numpy as np
import tensorflow as tf
import torch
from absl.testing import parameterized

from keras.src import backend
from keras.src import testing
from keras.src.distribution import distribution_lib as dist_lib
from keras.src.testing.test_utils import named_product
from keras.src.trainers.data_adapters.torch_data_loader_adapter import (
    TorchDataLoaderAdapter,
)


class TestTorchDataLoaderAdapter(testing.TestCase):
    def test_basic_dataloader(self):
        x = torch.normal(2, 3, size=(34, 4))
        y = torch.normal(1, 3, size=(34, 2))
        ds = torch.utils.data.TensorDataset(x, y)
        dataloader = torch.utils.data.DataLoader(ds, batch_size=16)
        adapter = TorchDataLoaderAdapter(dataloader)

        self.assertEqual(adapter.num_batches, 3)
        self.assertEqual(adapter.batch_size, 16)
        self.assertEqual(adapter.has_partial_batch, True)
        self.assertEqual(adapter.partial_batch_size, 2)

        if backend.backend() == "tensorflow":
            it = adapter.get_tf_dataset()
            expected_class = tf.Tensor
        elif backend.backend() == "jax":
            it = adapter.get_jax_iterator()
            expected_class = np.ndarray
        elif backend.backend() == "torch":
            it = adapter.get_torch_dataloader()
            expected_class = torch.Tensor
        else:
            it = adapter.get_numpy_iterator()
            expected_class = np.ndarray

        for i, batch in enumerate(it):
            self.assertEqual(len(batch), 2)
            bx, by = batch
            self.assertIsInstance(bx, expected_class)
            self.assertIsInstance(by, expected_class)
            self.assertEqual(bx.dtype, by.dtype)
            self.assertContainsExactSubsequence(str(bx.dtype), "float32")
            if i < 2:
                self.assertEqual(bx.shape, (16, 4))
                self.assertEqual(by.shape, (16, 2))
            else:
                self.assertEqual(bx.shape, (2, 4))
                self.assertEqual(by.shape, (2, 2))

    @parameterized.named_parameters(
        named_product(batch_size=[None, 3], implements_len=[True, False])
    )
    def test_dataloader_iterable_dataset(self, batch_size, implements_len):
        class TestIterableDataset(torch.utils.data.IterableDataset):
            def __init__(self):
                self.x = torch.normal(2, 3, size=(16, 4))
                self.y = torch.normal(1, 3, size=(16, 2))

            def __iter__(self):
                for _ in range(10):
                    yield (self.x, self.y)

        class TestIterableDatasetWithLen(TestIterableDataset):
            def __len__(self):
                return 10

        ds = (
            TestIterableDatasetWithLen()
            if implements_len
            else TestIterableDataset()
        )
        dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size)
        adapter = TorchDataLoaderAdapter(dataloader)

        if implements_len and batch_size:
            self.assertEqual(adapter.num_batches, math.ceil(10 / batch_size))
            self.assertEqual(adapter.batch_size, batch_size)
            self.assertEqual(adapter.has_partial_batch, True)
            self.assertEqual(adapter.partial_batch_size, 10 % batch_size)
        elif implements_len:
            self.assertEqual(adapter.num_batches, 10)
            self.assertEqual(adapter.batch_size, None)
            self.assertEqual(adapter.has_partial_batch, None)
            self.assertEqual(adapter.partial_batch_size, None)
        else:
            self.assertIsNone(adapter.num_batches)
            self.assertEqual(adapter.batch_size, batch_size)
            self.assertIsNone(adapter.has_partial_batch)
            self.assertIsNone(adapter.partial_batch_size)

        if backend.backend() == "tensorflow":
            it = adapter.get_tf_dataset()
            expected_class = tf.Tensor
        elif backend.backend() == "jax":
            it = adapter.get_jax_iterator()
            expected_class = np.ndarray
        elif backend.backend() == "torch":
            it = adapter.get_torch_dataloader()
            expected_class = torch.Tensor
        else:
            it = adapter.get_numpy_iterator()
            expected_class = np.ndarray

        batch_count = 0
        for i, batch in enumerate(it):
            batch_count += 1
            self.assertEqual(len(batch), 2)
            bx, by = batch
            self.assertIsInstance(bx, expected_class)
            self.assertIsInstance(by, expected_class)
            self.assertEqual(bx.dtype, by.dtype)
            self.assertContainsExactSubsequence(str(bx.dtype), "float32")
            if batch_size:
                if i < 3:
                    self.assertEqual(bx.shape, (batch_size, 16, 4))
                    self.assertEqual(by.shape, (batch_size, 16, 2))
                else:
                    self.assertEqual(bx.shape, (10 % batch_size, 16, 4))
                    self.assertEqual(by.shape, (10 % batch_size, 16, 2))
            else:
                self.assertEqual(bx.shape, (16, 4))
                self.assertEqual(by.shape, (16, 2))

        if batch_size:
            self.assertEqual(batch_count, math.ceil(10 / batch_size))
        else:
            self.assertEqual(batch_count, 10)

    def test_with_different_shapes(self):
        x = (
            [np.ones([4], "float32")] * 16
            + [np.ones([5], "float32")] * 16
            + [np.ones([6], "float32")] * 2
        )
        y = np.ones((34, 2), "float32")
        ds = torch.utils.data.StackDataset(x, y)
        dataloader = torch.utils.data.DataLoader(ds, batch_size=16)
        adapter = TorchDataLoaderAdapter(dataloader)

        self.assertEqual(adapter.num_batches, 3)
        self.assertEqual(adapter.batch_size, 16)
        self.assertEqual(adapter.has_partial_batch, True)
        self.assertEqual(adapter.partial_batch_size, 2)

        if backend.backend() == "tensorflow":
            it = adapter.get_tf_dataset()
        elif backend.backend() == "jax":
            it = adapter.get_jax_iterator()
        elif backend.backend() == "torch":
            it = adapter.get_torch_dataloader()
        else:
            it = adapter.get_numpy_iterator()

        for i, batch in enumerate(it):
            self.assertEqual(len(batch), 2)
            bx, by = batch
            self.assertEqual(bx.dtype, by.dtype)
            self.assertContainsExactSubsequence(str(bx.dtype), "float32")
            if i == 0:
                self.assertEqual(bx.shape, (16, 4))
                self.assertEqual(by.shape, (16, 2))
            elif i == 1:
                self.assertEqual(bx.shape, (16, 5))
                self.assertEqual(by.shape, (16, 2))
            else:
                self.assertEqual(bx.shape, (2, 6))
                self.assertEqual(by.shape, (2, 2))

    @patch("torch.distributed.is_available")
    @patch("torch.distributed.get_world_size")
    @patch("torch.distributed.get_rank")
    @patch("keras.src.backend.distribution_lib.num_processes", create=True)
    @patch("keras.src.backend.distribution_lib.process_id", create=True)
    @patch("keras.src.distribution.distribution_lib.distribution")
    def test_dataparallel_sharding(self, *args):
        mock_distribution = args[0]
        mock_process_id = args[1]
        mock_num_processes = args[2]
        mock_get_rank = args[3]
        mock_get_world_size = args[4]
        mock_is_available = args[5]

        mock_is_available.return_value = True
        mock_get_world_size.return_value = 4
        mock_get_rank.return_value = 1
        mock_num_processes.return_value = 4
        mock_process_id.return_value = 1

        dist = dist_lib.DataParallel(
            devices=["cpu:0", "cpu:1", "cpu:2", "cpu:3"]
        )
        dist.auto_shard_dataset = True
        mock_distribution.return_value = dist

        x = torch.randn(100, 10)
        y = torch.randn(100, 1)
        dataset = torch.utils.data.TensorDataset(x, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=10)

        adapter = TorchDataLoaderAdapter(dataloader)
        new_dataloader = adapter.get_torch_dataloader()

        self.assertIsInstance(
            new_dataloader.sampler,
            torch.utils.data.distributed.DistributedSampler,
        )
        self.assertEqual(new_dataloader.sampler.num_replicas, 4)
        self.assertEqual(new_dataloader.sampler.rank, 1)

    @patch("torch.distributed.is_available")
    @patch("torch.distributed.get_world_size")
    @patch("torch.distributed.get_rank")
    @patch("keras.src.backend.distribution_lib.num_processes", create=True)
    @patch("keras.src.backend.distribution_lib.process_id", create=True)
    @patch("keras.src.distribution.distribution_lib.distribution")
    def test_modelparallel_sharding(self, *args):
        mock_distribution = args[0]
        mock_process_id = args[1]
        mock_num_processes = args[2]
        mock_get_rank = args[3]
        mock_get_world_size = args[4]
        mock_is_available = args[5]

        mock_is_available.return_value = True
        mock_get_world_size.return_value = 8
        mock_get_rank.return_value = 5
        mock_num_processes.return_value = 8
        mock_process_id.return_value = 5

        device_mesh = dist_lib.DeviceMesh(
            shape=(2, 4), axis_names=("data", "model"), devices=["cpu:0"] * 8
        )

        dist = dist_lib.ModelParallel(
            device_mesh=device_mesh,
            layout_map=dist_lib.LayoutMap(device_mesh),
            batch_dim_name="data",
        )
        dist.auto_shard_dataset = True
        mock_distribution.return_value = dist

        x = torch.randn(100, 10)
        y = torch.randn(100, 1)
        dataset = torch.utils.data.TensorDataset(x, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=10)

        adapter = TorchDataLoaderAdapter(dataloader)
        new_dataloader = adapter.get_torch_dataloader()

        self.assertIsInstance(
            new_dataloader.sampler,
            torch.utils.data.distributed.DistributedSampler,
        )
        # num_model_replicas = dist.device_mesh.shape[0] = 2
        # num_process = 8, process_id = 5
        # num_model_replicas < num_process:
        # num_replicas = num_model_replicas = 2
        # processes_per_replica = 8 // 2 = 4
        # rank = 5 // 4 = 1
        self.assertEqual(new_dataloader.sampler.num_replicas, 2)
        self.assertEqual(new_dataloader.sampler.rank, 1)

    @patch("torch.distributed.is_available")
    @patch("torch.distributed.get_world_size")
    @patch("torch.distributed.get_rank")
    @patch("keras.src.backend.distribution_lib.num_processes", create=True)
    @patch("keras.src.backend.distribution_lib.process_id", create=True)
    @patch("keras.src.distribution.distribution_lib.distribution")
    def test_modelparallel_sharding_large_mesh(self, *args):
        mock_distribution = args[0]
        mock_process_id = args[1]
        mock_num_processes = args[2]
        mock_get_rank = args[3]
        mock_get_world_size = args[4]
        mock_is_available = args[5]

        mock_is_available.return_value = True
        mock_get_world_size.return_value = 4
        mock_get_rank.return_value = 2
        mock_num_processes.return_value = 4
        mock_process_id.return_value = 2

        device_mesh = dist_lib.DeviceMesh(
            shape=(8, 2), axis_names=("data", "model"), devices=["cpu:0"] * 16
        )

        dist = dist_lib.ModelParallel(
            device_mesh=device_mesh,
            layout_map=dist_lib.LayoutMap(device_mesh),
            batch_dim_name="data",
        )
        dist.auto_shard_dataset = True
        mock_distribution.return_value = dist

        x = torch.randn(100, 10)
        y = torch.randn(100, 1)
        dataset = torch.utils.data.TensorDataset(x, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=10)

        adapter = TorchDataLoaderAdapter(dataloader)
        new_dataloader = adapter.get_torch_dataloader()

        self.assertIsInstance(
            new_dataloader.sampler,
            torch.utils.data.distributed.DistributedSampler,
        )
        # num_model_replicas = 8
        # num_process = 4, process_id = 2
        # num_model_replicas >= num_process:
        # num_replicas = num_process = 4
        # rank = process_id = 2
        self.assertEqual(new_dataloader.sampler.num_replicas, 4)
        self.assertEqual(new_dataloader.sampler.rank, 2)
