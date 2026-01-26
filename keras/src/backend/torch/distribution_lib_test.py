"""Tests for torch distribution_lib.py."""

import os
import pytest

# Set torch backend before any keras imports
os.environ["KERAS_BACKEND"] = "torch"

import torch
import numpy as np

from keras.src import testing
from keras.src.backend.torch import distribution_lib


class TestListDevices(testing.TestCase):
    def test_list_devices_gpu(self):
        """Test listing GPU devices."""
        devices = distribution_lib.list_devices("gpu")
        if torch.cuda.is_available():
            self.assertEqual(len(devices), torch.cuda.device_count())
            for d in devices:
                self.assertTrue(d.startswith("cuda:"))
        else:
            self.assertEqual(devices, ["cpu:0"])

    def test_list_devices_cpu(self):
        """Test listing CPU devices."""
        devices = distribution_lib.list_devices("cpu")
        self.assertEqual(devices, ["cpu:0"])

    def test_list_devices_default(self):
        """Test default device listing."""
        devices = distribution_lib.list_devices()
        if torch.cuda.is_available():
            self.assertEqual(len(devices), torch.cuda.device_count())
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            self.assertGreater(len(devices), 0)
        elif torch.backends.mps.is_available():
            self.assertEqual(devices, ["mps:0"])
        else:
            self.assertEqual(devices, ["cpu:0"])


class TestGetDeviceCount(testing.TestCase):
    def test_get_device_count_gpu(self):
        """Test counting GPU devices."""
        count = distribution_lib.get_device_count("gpu")
        if torch.cuda.is_available():
            self.assertEqual(count, torch.cuda.device_count())
        else:
            self.assertEqual(count, 0)

    def test_get_device_count_cpu(self):
        """Test counting CPU devices."""
        count = distribution_lib.get_device_count("cpu")
        self.assertEqual(count, 1)

    def test_get_device_count_default(self):
        """Test default device counting."""
        count = distribution_lib.get_device_count()
        if torch.cuda.is_available():
            self.assertEqual(count, torch.cuda.device_count())
        else:
            self.assertEqual(count, 0)


class TestDistributeVariable(testing.TestCase):
    def test_distribute_variable_numpy_array(self):
        """Test distributing numpy array."""
        arr = np.random.randn(10, 20)
        result = distribution_lib.distribute_variable(arr, None)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (10, 20))

    def test_distribute_variable_torch_tensor(self):
        """Test distributing torch tensor."""
        tensor = torch.randn(10, 20)
        result = distribution_lib.distribute_variable(tensor, None)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (10, 20))


class TestDistributeTensor(testing.TestCase):
    def test_distribute_tensor(self):
        """Test distributing tensor."""
        tensor = torch.randn(10, 20)
        result = distribution_lib.distribute_tensor(tensor, None)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (10, 20))


class TestDistributedUtilities(testing.TestCase):
    def test_is_distributed(self):
        """Test checking if distributed is initialized."""
        # Should return False by default
        result = distribution_lib.is_distributed()
        self.assertFalse(result)

    def test_get_local_rank(self):
        """Test getting local rank."""
        rank = distribution_lib.get_local_rank()
        self.assertEqual(rank, 0)

    def test_get_world_size(self):
        """Test getting world size."""
        size = distribution_lib.get_world_size()
        self.assertEqual(size, 1)

    def test_get_rank(self):
        """Test getting rank."""
        rank = distribution_lib.get_rank()
        self.assertEqual(rank, 0)


class TestDataParallelUtilities(testing.TestCase):
    def test_create_data_parallel_model_single_gpu(self):
        """Test creating data parallel model with single GPU."""
        from keras.src import layers
        from keras.src import Model

        inputs = layers.Input(shape=(10,))
        outputs = layers.Dense(5)(inputs)
        model = Model(inputs, outputs)

        # With single GPU, should return same model
        wrapped = distribution_lib.create_data_parallel_model(model)
        self.assertEqual(wrapped, model)

    def test_create_data_parallel_model_multi_gpu(self):
        """Test creating data parallel model with multiple GPUs."""
        if torch.cuda.device_count() < 2:
            self.skipTest("Need at least 2 GPUs for this test")

        from keras.src import layers
        from keras.src import Model

        inputs = layers.Input(shape=(10,))
        outputs = layers.Dense(5)(inputs)
        model = Model(inputs, outputs)

        wrapped = distribution_lib.create_data_parallel_model(model)
        self.assertIsInstance(wrapped, torch.nn.DataParallel)


class TestDistributedSampler(testing.TestCase):
    def test_create_distributed_sampler(self):
        """Test creating distributed sampler."""
        dataset = torch.utils.data.TensorDataset(
            torch.randn(100, 10),
            torch.randint(0, 10, (100,))
        )

        sampler = distribution_lib.create_distributed_sampler(
            dataset,
            num_replicas=1,
            rank=0,
        )

        self.assertIsInstance(sampler, torch.utils.data.distributed.DistributedSampler)

    def test_create_dataloader(self):
        """Test creating dataloader."""
        dataset = torch.utils.data.TensorDataset(
            torch.randn(100, 10),
            torch.randint(0, 10, (100,))
        )

        dataloader = distribution_lib.create_dataloader(
            dataset,
            batch_size=10,
            shuffle=False,
        )

        self.assertIsInstance(dataloader, torch.utils.data.DataLoader)


class TestDeviceUtilities(testing.TestCase):
    def test_get_default_device(self):
        """Test getting default device."""
        device = distribution_lib.get_default_device()
        if torch.cuda.is_available():
            self.assertEqual(device.type, "cuda")
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            self.assertEqual(device.type, "xpu")
        elif torch.backends.mps.is_available():
            self.assertEqual(device.type, "mps")
        else:
            self.assertEqual(device.type, "cpu")

    def test_get_current_device(self):
        """Test getting current device."""
        device = distribution_lib.get_current_device()
        self.assertIsInstance(device, torch.device)

    def test_move_model_to_device(self):
        """Test moving model to device."""
        from keras.src import layers
        from keras.src import Model

        inputs = layers.Input(shape=(10,))
        outputs = layers.Dense(5)(inputs)
        model = Model(inputs, outputs)

        device = distribution_lib.get_default_device()
        moved = distribution_lib.move_model_to_device(model, device)

        # Model should be wrapped if it's a torch Module
        if isinstance(moved, torch.nn.Module):
            # Check if parameters are on correct device
            for param in moved.parameters():
                self.assertEqual(param.device, device)


class TestGradientUtilities(testing.TestCase):
    def test_get_gradients(self):
        """Test getting gradients from model."""
        from keras.src import layers
        from keras.src import Model

        inputs = layers.Input(shape=(10,))
        outputs = layers.Dense(5)(inputs)
        model = Model(inputs, outputs)

        # Forward pass
        x = torch.randn(5, 10)
        y = model(x)
        loss = y.sum()

        # Backward pass
        loss.backward()

        # Get gradients
        gradients = distribution_lib.get_gradients(model)
        self.assertIsInstance(gradients, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

