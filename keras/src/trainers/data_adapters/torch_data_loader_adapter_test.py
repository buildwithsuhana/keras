import math
import os
from unittest.mock import patch

import numpy as np
import pytest
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
    def test_basic_flow(self):
        x = torch.randn(10, 4)
        y = torch.randn(10, 2)
        dataset = torch.utils.data.TensorDataset(x, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)

        adapter = TorchDataLoaderAdapter(dataloader)
        self.assertEqual(adapter.num_batches, 5)
        self.assertEqual(adapter.batch_size, 2)
        self.assertEqual(adapter.has_partial_batch, False)

        it = adapter.get_numpy_iterator()
        for i, batch in enumerate(it):
            self.assertEqual(len(batch), 2)
            bx, by = batch
            self.assertAllClose(bx, x[i * 2 : (i + 1) * 2].numpy())
            self.assertAllClose(by, y[i * 2 : (i + 1) * 2].numpy())

    def test_get_tf_dataset(self):
        x = torch.randn(10, 4)
        y = torch.randn(10, 2)
        dataset = torch.utils.data.TensorDataset(x, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)

        adapter = TorchDataLoaderAdapter(dataloader)
        ds = adapter.get_tf_dataset()
        for i, batch in enumerate(ds):
            self.assertEqual(len(batch), 2)
            bx, by = batch
            self.assertAllClose(bx, x[i * 2 : (i + 1) * 2].numpy())
            self.assertAllClose(by, y[i * 2 : (i + 1) * 2].numpy())

    @parameterized.named_parameters(
        named_product(
            list_inputs=[False, True],
        )
    )
    def test_multi_inputs_and_outputs(self, list_inputs):
        x1 = torch.randn(10, 4)
        x2 = torch.randn(10, 5)
        y1 = torch.randn(10, 2)
        y2 = torch.randn(10, 3)
        if list_inputs:
            x = [x1, x2]
            y = [y1, y2]
        else:
            x = {"x1": x1, "x2": x2}
            y = {"y1": y1, "y2": y2}

        class MultiInputDataset(torch.utils.data.Dataset):
            def __init__(self, x, y):
                self.x = x
                self.y = y

            def __len__(self):
                return 10

            def __getitem__(self, index):
                if isinstance(self.x, list):
                    xi = [ti[index] for ti in self.x]
                    yi = [ti[index] for ti in self.y]
                else:
                    xi = {k: v[index] for k, v in self.x.items()}
                    yi = {k: v[index] for k, v in self.y.items()}
                return xi, yi

        dataset = MultiInputDataset(x, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)

        adapter = TorchDataLoaderAdapter(dataloader)
        self.assertEqual(adapter.num_batches, 5)
        self.assertEqual(adapter.batch_size, 2)

        it = adapter.get_numpy_iterator()
        for i, batch in enumerate(it):
            self.assertEqual(len(batch), 2)
            bx, by = batch
            if list_inputs:
                self.assertIsInstance(bx, list)
                self.assertIsInstance(by, list)
                self.assertAllClose(bx[0], x1[i * 2 : (i + 1) * 2].numpy())
                self.assertAllClose(bx[1], x2[i * 2 : (i + 1) * 2].numpy())
                self.assertAllClose(by[0], y1[i * 2 : (i + 1) * 2].numpy())
                self.assertAllClose(by[1], y2[i * 2 : (i + 1) * 2].numpy())
            else:
                self.assertIsInstance(bx, dict)
                self.assertIsInstance(by, dict)
                self.assertAllClose(bx["x1"], x1[i * 2 : (i + 1) * 2].numpy())
                self.assertAllClose(bx["x2"], x2[i * 2 : (i + 1) * 2].numpy())
                self.assertAllClose(by["y1"], y1[i * 2 : (i + 1) * 2].numpy())
                self.assertAllClose(by["y2"], y2[i * 2 : (i + 1) * 2].numpy())

        ds = adapter.get_tf_dataset()
        for i, batch in enumerate(ds):
            self.assertEqual(len(batch), 2)
            bx, by = batch
            if list_inputs:
                self.assertIsInstance(bx, tuple)
                self.assertIsInstance(by, tuple)
                self.assertAllClose(bx[0], x1[i * 2 : (i + 1) * 2].numpy())
                self.assertAllClose(bx[1], x2[i * 2 : (i + 1) * 2].numpy())
                self.assertAllClose(by[0], y1[i * 2 : (i + 1) * 2].numpy())
                self.assertAllClose(by[1], y2[i * 2 : (i + 1) * 2].numpy())
            else:
                self.assertIsInstance(bx, dict)
                self.assertIsInstance(by, dict)
                self.assertAllClose(bx["x1"], x1[i * 2 : (i + 1) * 2].numpy())
                self.assertAllClose(bx["x2"], x2[i * 2 : (i + 1) * 2].numpy())
                self.assertAllClose(by["y1"], y1[i * 2 : (i + 1) * 2].numpy())
                self.assertAllClose(by["y2"], y2[i * 2 : (i + 1) * 2].numpy())

    def test_partial_batches(self):
        x = torch.randn(10, 4)
        y = torch.randn(10, 2)
        dataset = torch.utils.data.TensorDataset(x, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=3)

        adapter = TorchDataLoaderAdapter(dataloader)
        self.assertEqual(adapter.num_batches, 4)
        self.assertEqual(adapter.batch_size, 3)
        self.assertEqual(adapter.has_partial_batch, True)
        self.assertEqual(adapter.partial_batch_size, 1)

        it = adapter.get_numpy_iterator()
        for i, batch in enumerate(it):
            bx, by = batch
            if i < 3:
                self.assertEqual(bx.shape, (3, 4))
                self.assertEqual(by.shape, (3, 2))
            else:
                self.assertEqual(bx.shape, (1, 4))
                self.assertEqual(by.shape, (1, 2))

    def test_invalid_dataloader(self):
        with self.assertRaisesRegex(ValueError, "Expected argument `dataloader`"):
            TorchDataLoaderAdapter([1, 2, 3])

    def test_rebatch_tf_dataset(self):
        # TorchDataLoaderAdapter should return a tf.data.Dataset with the correct
        # batch size even if it was rebatched.
        x = torch.randn(16, 6)
        y = torch.randn(16, 2)
        dataset = torch.utils.data.TensorDataset(x, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4)

        adapter = TorchDataLoaderAdapter(dataloader)
        ds = adapter.get_tf_dataset()
        self.assertEqual(adapter.batch_size, 4)

        for i, batch in enumerate(ds):
            bx, by = batch
            if i < 4:
                self.assertEqual(bx.shape, (4, 6))
                self.assertEqual(by.shape, (4, 2))
            else:
                self.assertEqual(bx.shape, (2, 6))
                self.assertEqual(by.shape, (2, 2))

    @parameterized.named_parameters(
        ("dataparallel", "dp", 4, 1, (4,), 4, 1),
        ("modelparallel", "mp", 8, 5, (2, 4), 2, 1),
        ("modelparallel_large_mesh", "mp", 4, 2, (8, 2), 4, 2),
    )
    @patch("torch.distributed.is_available")
    @patch("torch.distributed.get_world_size")
    @patch("torch.distributed.get_rank")
    @patch("keras.src.distribution.distribution_lib.distribution_lib")
    @patch("keras.src.distribution.distribution_lib.distribution")
    def test_sharding(
        self,
        dist_type,
        world_size,
        rank,
        mesh_shape,
        expected_num_replicas,
        expected_rank,
        mock_distribution,
        mock_backend_dist_lib,
        mock_get_rank,
        mock_get_world_size,
        mock_is_available,
    ):
        mock_is_available.return_value = True
        mock_get_world_size.return_value = world_size
        mock_get_rank.return_value = rank
        mock_backend_dist_lib.num_processes.return_value = world_size
        mock_backend_dist_lib.process_id.return_value = rank

        if dist_type == "dp":
            dist = dist_lib.DataParallel(devices=["cpu:0"] * world_size)
        else:
            device_mesh = dist_lib.DeviceMesh(
                shape=mesh_shape,
                axis_names=("data", "model"),
                devices=["cpu:0"] * np.prod(mesh_shape),
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
        self.assertEqual(
            new_dataloader.sampler.num_replicas, expected_num_replicas
        )
        self.assertEqual(new_dataloader.sampler.rank, expected_rank)

    @parameterized.named_parameters(
        ("dataparallel", "dp", 4, 1, (4,), 4, 1),
        ("modelparallel", "mp", 8, 5, (2, 4), 2, 1),
    )
    @patch("torch.distributed.is_available")
    @patch("torch.distributed.get_world_size")
    @patch("torch.distributed.get_rank")
    @patch("keras.src.distribution.distribution_lib.distribution_lib")
    @patch("keras.src.distribution.distribution_lib.distribution")
    def test_sharding_iterable_dataset(
        self,
        dist_type,
        world_size,
        rank,
        mesh_shape,
        expected_num_replicas,
        expected_rank,
        mock_distribution,
        mock_backend_dist_lib,
        mock_get_rank,
        mock_get_world_size,
        mock_is_available,
    ):
        mock_is_available.return_value = True
        mock_get_world_size.return_value = world_size
        mock_get_rank.return_value = rank
        mock_backend_dist_lib.num_processes.return_value = world_size
        mock_backend_dist_lib.process_id.return_value = rank

        if dist_type == "dp":
            dist = dist_lib.DataParallel(devices=["cpu:0"] * world_size)
        else:
            device_mesh = dist_lib.DeviceMesh(
                shape=mesh_shape,
                axis_names=("data", "model"),
                devices=["cpu:0"] * np.prod(mesh_shape),
            )
            dist = dist_lib.ModelParallel(
                device_mesh=device_mesh,
                layout_map=dist_lib.LayoutMap(device_mesh),
                batch_dim_name="data",
            )
        dist.auto_shard_dataset = True
        mock_distribution.return_value = dist

        class TestIterableDataset(torch.utils.data.IterableDataset):
            def __iter__(self):
                for i in range(100):
                    yield torch.tensor([i])

            def __len__(self):
                return 100

        dataset = TestIterableDataset()
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=10)

        adapter = TorchDataLoaderAdapter(dataloader)
        new_dataloader = adapter.get_torch_dataloader()

        self.assertTrue(
            "ShardedIterableDataset" in str(type(new_dataloader.dataset))
        )
        self.assertEqual(
            new_dataloader.dataset.num_replicas, expected_num_replicas
        )
        self.assertEqual(new_dataloader.dataset.rank, expected_rank)

        items = list(new_dataloader)
        # 100 items total, 10 items per batch.
        # Each replica should get 100 / expected_num_replicas items.
        expected_items = 100 // expected_num_replicas
        self.assertEqual(sum(len(b) for b in items), expected_items)

        # Check that we are getting the correct items (interleaved sharding)
        first_item = next(iter(new_dataloader))[0]
        self.assertEqual(first_item.item(), expected_rank)

    @pytest.mark.skipif(
        backend.backend() != "torch",
        reason="Only for torch backend",
    )
    def test_torch_dataloader_distribute_integration(self):
        # Initialize torch distributed
        if not torch.distributed.is_initialized():
            os.environ["WORLD_SIZE"] = "1"
            os.environ["RANK"] = "0"
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "12356"
            torch.distributed.init_process_group(
                backend="gloo",
                rank=0,
                world_size=1,
            )

        from keras.src.distribution import distribution_lib

        dp = distribution_lib.DataParallel(devices=["cpu:0"])
        # Force multi-process to test sampler wiring even with 1 process
        dp._is_multi_process = True
        with dp.scope():
            x = torch.randn(10, 2)
            y = torch.randn(10, 1)
            dataset = torch.utils.data.TensorDataset(x, y)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)

            adapter = TorchDataLoaderAdapter(dataloader)
            new_dataloader = adapter.get_torch_dataloader()

            self.assertIsInstance(
                new_dataloader.sampler,
                torch.utils.data.distributed.DistributedSampler,
            )
            self.assertEqual(new_dataloader.sampler.num_replicas, 1)
            self.assertEqual(new_dataloader.sampler.rank, 0)
