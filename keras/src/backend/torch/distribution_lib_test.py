import os
import socket
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp

from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src import testing
from keras.src.backend.torch import distribution_lib as backend_dlib
from keras.src.distribution import distribution_lib


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.mark.skipif(
    backend.backend() != "torch", reason="Only for Torch backend"
)
class TorchDistributedTestCase(testing.TestCase):
    def setUp(self):
        super().setUp()
        self.world_size = 2

    @staticmethod
    def _worker_wrapper(rank, world_size, port, test_fn, cls):
        import torch.distributed as dist

        os.environ.update(
            {
                "MASTER_ADDR": "localhost",
                "MASTER_PORT": str(port),
                "RANK": str(rank),
                "WORLD_SIZE": str(world_size),
                "LOCAL_RANK": str(rank),
                "KERAS_TORCH_DEVICE": "cpu",
                "PYTORCH_ENABLE_MPS_FALLBACK": "1",
            }
        )
        try:
            dist.init_process_group("gloo", rank=rank, world_size=world_size)
            test_fn(cls("setUp"), rank, world_size)
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()

    def run_distributed(self, test_fn, world_size=None):
        previous_device = os.environ.get("KERAS_TORCH_DEVICE")
        os.environ["KERAS_TORCH_DEVICE"] = "cpu"
        world_size = world_size or self.world_size
        port = find_free_port()
        try:
            mp.spawn(
                TorchDistributedTestCase._worker_wrapper,
                args=(world_size, port, test_fn, self.__class__),
                nprocs=world_size,
                join=True,
            )
        finally:
            if previous_device is None:
                del os.environ["KERAS_TORCH_DEVICE"]
            else:
                os.environ["KERAS_TORCH_DEVICE"] = previous_device


class TorchDeviceDiscoveryTest(TorchDistributedTestCase):
    @staticmethod
    def _list_devices_test(self, rank, world_size):
        # Test default (None -> gpu)
        devices = distribution_lib.list_devices()
        self.assertLen(devices, world_size)
        for i in range(world_size):
            self.assertEqual(devices[i], f"gpu:{i}")

        # Test explicit device type
        devices = distribution_lib.list_devices("cpu")
        self.assertLen(devices, world_size)
        for i in range(world_size):
            self.assertEqual(devices[i], f"cpu:{i}")

    @staticmethod
    def _get_device_count_test(self, rank, world_size):
        self.assertEqual(backend_dlib.get_device_count(), world_size)

    @staticmethod
    def _list_devices_default_test(self, rank, world_size):
        # Test default device_type = None -> "gpu"
        devices = backend_dlib.list_devices(None)
        self.assertLen(devices, world_size)
        self.assertTrue(all("gpu" in d for d in devices))

    def test_list_devices(self):
        self.run_distributed(TorchDeviceDiscoveryTest._list_devices_test)

    def test_list_devices_default(self):
        self.run_distributed(
            TorchDeviceDiscoveryTest._list_devices_default_test
        )

    def test_get_device_count(self):
        self.run_distributed(TorchDeviceDiscoveryTest._get_device_count_test)

    def test_get_device_count_fallback(self):
        from unittest.mock import patch

        # Case 1: is_initialized is False, WORLD_SIZE in environ
        with patch("torch.distributed.is_initialized", return_value=False):
            with patch.dict(os.environ, {"WORLD_SIZE": "4"}):
                self.assertEqual(backend_dlib.get_device_count(), 4)

        # Case 2: is_initialized is False, WORLD_SIZE NOT in environ
        with patch("torch.distributed.is_initialized", return_value=False):
            with patch.dict(os.environ, {}):
                if "WORLD_SIZE" in os.environ:
                    del os.environ["WORLD_SIZE"]
                expected = torch.cuda.device_count() or 1
                self.assertEqual(backend_dlib.get_device_count(), expected)


class TorchProcessManagementTest(TorchDistributedTestCase):
    @staticmethod
    def _num_processes_test(self, rank, world_size):
        self.assertEqual(backend_dlib.num_processes(), world_size)

    @staticmethod
    def _process_id_test(self, rank, world_size):
        self.assertEqual(backend_dlib.process_id(), rank)

    def test_num_processes(self):
        self.run_distributed(TorchProcessManagementTest._num_processes_test)

    def test_num_processes_fallback(self):
        with patch("torch.distributed.is_initialized", return_value=False):
            self.assertEqual(backend_dlib.num_processes(), 1)

    def test_process_id(self):
        self.run_distributed(TorchProcessManagementTest._process_id_test)

    def test_process_id_fallback(self):
        with patch("torch.distributed.is_initialized", return_value=False):
            self.assertEqual(backend_dlib.process_id(), 0)

    def test_initialize(self):
        from unittest.mock import patch

        # Case 1: CUDA not available
        with patch("torch.distributed.init_process_group") as mock_init:
            with patch("torch.cuda.is_available", return_value=False):
                backend_dlib.initialize()
                mock_init.assert_called_once_with(backend="gloo")

        # Case 2: CUDA available
        with patch("torch.distributed.init_process_group") as mock_init:
            with patch("torch.cuda.is_available", return_value=True):
                with patch("torch.cuda.set_device") as mock_set_device:
                    with patch.dict(os.environ, {"LOCAL_RANK": "1"}):
                        backend_dlib.initialize()
                        mock_set_device.assert_called_once_with(1)
                        mock_init.assert_called_once_with(backend="nccl")


class TorchDeviceMeshMappingTest(TorchDistributedTestCase):
    @staticmethod
    def _to_backend_mesh_test(self, rank, world_size):
        from torch.distributed.device_mesh import DeviceMesh

        mesh = distribution_lib.DeviceMesh((world_size,), ["data"])
        torch_mesh = backend_dlib._to_backend_mesh(mesh)
        self.assertIsInstance(torch_mesh, DeviceMesh)
        self.assertEqual(torch_mesh.mesh.shape, (world_size,))
        self.assertEqual(torch_mesh.mesh_dim_names, ("data",))

    @staticmethod
    def _to_backend_layout_test(self, rank, world_size):
        from torch.distributed.tensor import Shard

        mesh = distribution_lib.DeviceMesh((world_size,), ["data"])
        layout = distribution_lib.TensorLayout(["data", None], mesh)
        backend_layout = backend_dlib._to_backend_layout(layout)
        placements = backend_layout.placements
        self.assertLen(placements, 1)
        self.assertIsInstance(placements[0], Shard)
        self.assertEqual(placements[0].dim, 0)

    def test_to_backend_mesh(self):
        self.run_distributed(TorchDeviceMeshMappingTest._to_backend_mesh_test)

    def test_to_backend_layout(self):
        self.run_distributed(TorchDeviceMeshMappingTest._to_backend_layout_test)


class TorchTensorDistributionTest(TorchDistributedTestCase):
    @staticmethod
    def _distribute_tensor_test(self, rank, world_size):
        from torch.distributed.tensor import DTensor

        mesh = distribution_lib.DeviceMesh((world_size,), ["data"])
        layout = distribution_lib.TensorLayout(["data", None], mesh)
        dist = distribution_lib.ModelParallel(
            device_mesh=mesh, layout_map=distribution_lib.LayoutMap(mesh)
        )
        with dist.scope():
            tensor = backend_dlib.distribute_tensor(
                torch.randn(world_size * 2, 4), layout
            )
            self.assertIsInstance(tensor, DTensor)
            self.assertEqual(tensor.shape, (world_size * 2, 4))
            self.assertEqual(tensor.to_local().shape, (2, 4))

            # Test redistribution (DTensor -> DTensor with new layout)
            from torch.distributed.tensor import Replicate

            new_layout = distribution_lib.TensorLayout([None, None], mesh)
            redistributed = backend_dlib.distribute_tensor(tensor, new_layout)
            self.assertIsInstance(redistributed, DTensor)
            self.assertTrue(
                all(isinstance(p, Replicate) for p in redistributed.placements)
            )

    @staticmethod
    def _distribute_data_input_test(self, rank, world_size):
        from torch.distributed.tensor import DTensor

        mesh = distribution_lib.DeviceMesh((world_size,), ["batch"])
        layout = distribution_lib.TensorLayout(["batch", None], mesh)
        data = torch.arange(rank * 4, (rank + 1) * 4).reshape(2, 2).float()
        dist = distribution_lib.ModelParallel(
            device_mesh=mesh,
            layout_map=distribution_lib.LayoutMap(mesh),
            batch_dim_name="batch",
        )
        with dist.scope():
            tensor = backend_dlib.distribute_data_input(data, layout, "batch")
            self.assertIsInstance(tensor, DTensor)
            self.assertEqual(tensor.shape, (world_size * 2, 2))
            self.assertAllClose(tensor.to_local(), data)

    def test_distribute_tensor(self):
        self.run_distributed(
            TorchTensorDistributionTest._distribute_tensor_test
        )

    def test_distribute_data_input(self):
        self.run_distributed(
            TorchTensorDistributionTest._distribute_data_input_test
        )

    def _distribute_data_input_edge_cases_test(self, rank, world_size):
        from torch.distributed.tensor import DTensor

        mesh = distribution_lib.DeviceMesh((world_size,), ["batch"])
        layout = distribution_lib.TensorLayout(["batch", None], mesh)
        dist = distribution_lib.ModelParallel(
            layout_map=distribution_lib.LayoutMap(mesh),
            batch_dim_name="batch",
        )
        with dist.scope():
            # 1. DTensor input (should return as is)
            tensor = torch.randn(world_size * 2, 2)
            dtensor = backend_dlib.distribute_data_input(
                tensor, layout, "batch"
            )
            self.assertIs(
                backend_dlib.distribute_data_input(dtensor, layout, "batch"),
                dtensor,
            )

            # 2. Numpy input
            data_np = np.random.randn(2, 2).astype("float32")
            dtensor_np = backend_dlib.distribute_data_input(
                data_np, layout, "batch"
            )
            self.assertIsInstance(dtensor_np, DTensor)
            self.assertAllClose(dtensor_np.to_local(), data_np)

            # 3. Meta device tensor
            meta_tensor = torch.randn(2, 2, device="meta")
            output = backend_dlib.distribute_data_input(
                meta_tensor, layout, "batch"
            )
            self.assertEqual(output.device.type, "meta")

    def test_distribute_data_input_edge_cases(self):
        self.run_distributed(
            TorchTensorDistributionTest._distribute_data_input_edge_cases_test
        )

    def _distribute_tensor_none_cases_test(self, rank, world_size):
        tensor = torch.randn(2, 2)
        # 1. layout is None
        self.assertIs(backend_dlib.distribute_tensor(tensor, None), tensor)

        # 2. distribution is not ModelParallel (e.g. DataParallel)
        mesh = distribution_lib.DeviceMesh((world_size,), ["batch"])
        dist = distribution_lib.DataParallel(mesh)
        layout = distribution_lib.TensorLayout(["batch", None], mesh)
        with dist.scope():
            self.assertIs(
                backend_dlib.distribute_tensor(tensor, layout), tensor
            )

    def test_distribute_tensor_none_cases(self):
        self.run_distributed(
            TorchTensorDistributionTest._distribute_tensor_none_cases_test
        )


class TorchVariableDistributionAwarenessTest(TorchDistributedTestCase):
    @staticmethod
    def _variable_distribution_awareness_test(self, rank, world_size):
        from torch.distributed.tensor import DTensor

        mesh = distribution_lib.DeviceMesh((world_size,), ["model"])
        layout_map = distribution_lib.LayoutMap(mesh)
        layout_map[".*kernel"] = distribution_lib.TensorLayout([None, "model"])
        dist = distribution_lib.ModelParallel(
            layout_map=layout_map, batch_dim_name="model"
        )
        with dist.scope():
            layer = layers.Dense(world_size * 4)
            layer.build((8, 8))
        self.assertIsInstance(layer.kernel.value, DTensor)
        self.assertEqual(layer.kernel.value.to_local().shape, (8, 4))

    def test_variable_distribution_awareness(self):
        self.run_distributed(
            TorchVariableDistributionAwarenessTest._variable_distribution_awareness_test
        )


class TorchTrainerModeTest(TorchDistributedTestCase):
    @staticmethod
    def _on_batch_mode_test(self, rank, world_size):
        mesh = distribution_lib.DeviceMesh((world_size,), ["batch"])
        dist = distribution_lib.DataParallel(mesh, auto_shard_dataset=False)
        with dist.scope():
            model = models.Sequential(
                [layers.Input(shape=(8,)), layers.Dense(2)]
            )
            model.compile(optimizer="adam", loss="mse")

            # train_on_batch should set train mode
            model.train_on_batch(np.random.randn(2, 8), np.random.randn(2, 2))
            self.assertTrue(model._ddp_model.training)

            # test_on_batch should set eval mode
            model.test_on_batch(np.random.randn(2, 8), np.random.randn(2, 2))
            self.assertFalse(model._ddp_model.training)

            # train_on_batch should set train mode back
            model.train_on_batch(np.random.randn(2, 8), np.random.randn(2, 2))
            self.assertTrue(model._ddp_model.training)

            # predict_on_batch should set eval mode
            model.predict_on_batch(np.random.randn(2, 8))
            self.assertFalse(model._ddp_model.training)

    def test_on_batch_mode(self):
        self.run_distributed(TorchTrainerModeTest._on_batch_mode_test)

    def test_train_on_batch_class_weight(self):
        model = models.Sequential([layers.Dense(2)])
        model.compile(optimizer="adam", loss="mse")
        x = np.random.randn(4, 8).astype("float32")
        y = np.array([0, 1, 0, 1]).reshape(-1, 1).astype("float32")
        class_weight = {0: 1.0, 1: 2.0}
        logs = model.train_on_batch(
            x, y, class_weight=class_weight, return_dict=True
        )
        self.assertIn("loss", logs)

    def test_cached_eval_iterator(self):
        from keras.src.backend.torch.trainer import TorchEpochIterator

        model = models.Sequential([layers.Dense(2)])
        model.compile(optimizer="adam", loss="mse")
        x = np.random.randn(4, 8).astype("float32")
        y = np.random.randn(4, 2).astype("float32")

        # Manually create and set the iterator to test the cached path
        iterator = TorchEpochIterator(x=x, y=y, batch_size=2)
        model._eval_epoch_iterator = iterator

        # Call evaluate with cached iterator
        logs = model.evaluate(
            x, y, _use_cached_eval_dataset=True, return_dict=True
        )
        self.assertIn("loss", logs)
        # Ensure it was actually used (iterator was reset and consumed)
        self.assertIs(model._eval_epoch_iterator, iterator)


class TorchTrainerArchitectureTest(TorchDistributedTestCase):
    @staticmethod
    def _e2e_data_parallel_fit_test(self, rank, world_size):
        mesh = distribution_lib.DeviceMesh((world_size,), ["batch"])
        dist = distribution_lib.DataParallel(mesh, auto_shard_dataset=False)
        with dist.scope():
            model = models.Sequential(
                [layers.Input(shape=(8,)), layers.Dense(4), layers.Dense(2)]
            )
            model.compile(optimizer="adam", loss="mse", metrics=["mae"])
            x, y = np.random.randn(4, 8), np.random.randn(4, 2)
            model.fit(x, y, epochs=1, batch_size=2)
            self.assertNotEqual(len(model.evaluate(x, y, batch_size=2)), 0)

    @staticmethod
    def _e2e_model_parallel_fit_test(self, rank, world_size):
        mesh = distribution_lib.DeviceMesh((1, world_size), ["batch", "model"])
        layout_map = distribution_lib.LayoutMap(mesh)
        layout_map[".*dense.*kernel"] = distribution_lib.TensorLayout(
            [None, "model"]
        )
        dist = distribution_lib.ModelParallel(
            layout_map=layout_map, batch_dim_name="batch", auto_shard_dataset=0
        )
        with dist.scope():
            model = models.Sequential(
                [
                    layers.Input(shape=(8,)),
                    layers.Dense(world_size * 4),
                    layers.Dense(2),
                ]
            )
            model.compile(optimizer="adam", loss="mse")
            model.fit(
                np.random.randn(4, 8),
                np.random.randn(4, 2),
                epochs=1,
                batch_size=2,
            )

    @staticmethod
    def _keras_module_wrapper_test(self, rank, world_size):
        from keras.src.backend.torch.trainer import _KerasModuleWrapper

        # Test with model having both trainable and non-trainable weights
        inputs = layers.Input(shape=(8,))
        x = layers.Dense(4, name="dense_trainable")(inputs)  # trainable
        non_trainable_layer = layers.Dense(
            4, name="dense_non_trainable", trainable=False
        )
        x = non_trainable_layer(x)
        outputs = layers.Dense(2, name="dense_output")(x)
        model = models.Model(inputs, outputs)
        model.build(input_shape=(None, 8))

        # Verify splits non-empty for loop coverage
        trainable_weights = model.trainable_weights
        non_trainable_weights = model.non_trainable_weights
        self.assertGreater(len(trainable_weights), 0)
        self.assertGreater(len(non_trainable_weights), 0)

        wrapper = _KerasModuleWrapper(model)

        # Check parameters from trainable_weights (register_parameter)
        trainable_params = list(wrapper.parameters())
        self.assertGreater(len(trainable_params), 0)
        self.assertTrue(all(p.requires_grad for p in trainable_params))

        # Check buffers from non_trainable_weights (register_buffer)
        buffers = list(wrapper.buffers())
        self.assertGreater(len(buffers), 0)
        self.assertFalse(any(p.requires_grad for p in buffers))

        # Check forward with *args only
        x = torch.randn(2, 8)
        y = wrapper(x)
        self.assertEqual(y.shape, (2, 2))

        # Check forward with **kwargs (training=False)
        y_kwargs = wrapper(x, training=False)
        self.assertEqual(y_kwargs.shape, (2, 2))

        # Check forward with training=True arg
        y_train = wrapper(x, training=True)
        self.assertEqual(y_train.shape, (2, 2))

        # Original BN test for buffers like moving stats
        model_bn = models.Sequential(
            [
                layers.Input(shape=(8,)),
                layers.Dense(4),
                layers.BatchNormalization(),
                layers.Dense(2),
            ]
        )
        model_bn.build(input_shape=(None, 8))
        wrapper_bn = _KerasModuleWrapper(model_bn)
        self.assertLen(list(wrapper_bn.parameters()), 6)
        self.assertLen(list(wrapper_bn.buffers()), 2)

    def test_e2e_data_parallel_fit(self):
        self.run_distributed(
            TorchTrainerArchitectureTest._e2e_data_parallel_fit_test
        )

    def test_e2e_model_parallel_fit(self):
        self.run_distributed(
            TorchTrainerArchitectureTest._e2e_model_parallel_fit_test
        )

    def test_keras_module_wrapper(self):
        self.run_distributed(
            TorchTrainerArchitectureTest._keras_module_wrapper_test,
            world_size=1,
        )

    def test_should_torch_compile_warning(self):
        from unittest.mock import patch

        model = models.Sequential([layers.Dense(2)])
        model.compile(jit_compile=True)
        # Mock torch version < 2.1.0
        with patch("torch.__version__", "2.0.0"):
            with pytest.warns(
                UserWarning, match="Please upgrade to torch>=2.1.0"
            ):
                model._should_torch_compile()
                self.assertFalse(model.jit_compile)

    def test_steps_per_execution_error(self):
        model = models.Sequential([layers.Dense(2)])
        model.compile(steps_per_execution=2)
        with pytest.raises(ValueError, match="`steps_per_execution` must be 1"):
            model.make_train_function()


class TorchDataLoadingTest(TorchDistributedTestCase):
    @staticmethod
    def _dataloader_distributed_sampler_test(self, rank, world_size):
        from torch.utils.data import DataLoader
        from torch.utils.data import TensorDataset
        from torch.utils.data.distributed import DistributedSampler

        from keras.src.trainers.data_adapters.torch_data_loader_adapter import (
            TorchDataLoaderAdapter,
        )

        dataset = TensorDataset(
            torch.randn(world_size * 4, 8), torch.randn(world_size * 4, 2)
        )
        dataloader = DataLoader(dataset, batch_size=2)
        with distribution_lib.DataParallel().scope():
            new_dataloader = TorchDataLoaderAdapter(
                dataloader
            ).get_torch_dataloader()
            self.assertIsInstance(new_dataloader.sampler, DistributedSampler)
            self.assertEqual(new_dataloader.sampler.num_replicas, world_size)
            self.assertEqual(new_dataloader.sampler.rank, rank)

    @staticmethod
    def _dataloader_model_parallel_sampler_test(self, rank, world_size):
        from torch.utils.data import DataLoader
        from torch.utils.data import TensorDataset
        from torch.utils.data.distributed import DistributedSampler

        from keras.src.trainers.data_adapters.torch_data_loader_adapter import (
            TorchDataLoaderAdapter,
        )

        dataset = TensorDataset(
            torch.randn(world_size * 4, 8), torch.randn(world_size * 4, 2)
        )
        dataloader = DataLoader(dataset, batch_size=2)

        # 1. Test ModelParallel with batch sharding
        mesh = distribution_lib.DeviceMesh((world_size,), ["batch"])
        layout_map = distribution_lib.LayoutMap(mesh)
        dist = distribution_lib.ModelParallel(
            layout_map=layout_map, batch_dim_name="batch"
        )
        with dist.scope():
            new_dataloader = TorchDataLoaderAdapter(
                dataloader
            ).get_torch_dataloader()
            self.assertIsInstance(new_dataloader.sampler, DistributedSampler)
            self.assertEqual(new_dataloader.sampler.num_replicas, world_size)
            self.assertEqual(new_dataloader.sampler.rank, rank)

        # 2. Test ModelParallel without batch sharding (batch dim size 1)
        mesh2 = distribution_lib.DeviceMesh((1, world_size), ["batch", "model"])
        layout_map2 = distribution_lib.LayoutMap(mesh2)
        dist2 = distribution_lib.ModelParallel(
            layout_map=layout_map2, batch_dim_name="batch"
        )
        with dist2.scope():
            new_dataloader2 = TorchDataLoaderAdapter(
                dataloader
            ).get_torch_dataloader()
            self.assertNotIsInstance(
                new_dataloader2.sampler, DistributedSampler
            )

    @staticmethod
    def _dataloader_model_parallel_complex_sharding_test(
        self, rank, world_size
    ):
        from torch.utils.data import DataLoader
        from torch.utils.data import TensorDataset
        from torch.utils.data.distributed import DistributedSampler

        from keras.src.trainers.data_adapters.torch_data_loader_adapter import (
            TorchDataLoaderAdapter,
        )

        dataset = TensorDataset(torch.randn(16, 8), torch.randn(16, 2))
        # Test with shuffle=True to exercise the RandomSampler detection
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

        # 4 processes, mesh (2, 2), batch_dim_name='batch'
        # num_model_replicas (data parallel degree) = 2
        # num_process = 4
        # rank 0, 1 -> data rank 0
        # rank 2, 3 -> data rank 1
        mesh = distribution_lib.DeviceMesh((2, 2), ["batch", "model"])
        layout_map = distribution_lib.LayoutMap(mesh)
        dist = distribution_lib.ModelParallel(
            layout_map=layout_map, batch_dim_name="batch"
        )
        with dist.scope():
            new_dataloader = TorchDataLoaderAdapter(
                dataloader
            ).get_torch_dataloader()
            self.assertIsInstance(new_dataloader.sampler, DistributedSampler)
            self.assertEqual(new_dataloader.sampler.num_replicas, 2)
            self.assertEqual(new_dataloader.sampler.rank, rank // 2)
            self.assertTrue(new_dataloader.sampler.shuffle)

    @staticmethod
    def _dataloader_no_auto_shard_test(self, rank, world_size):
        from torch.utils.data import DataLoader
        from torch.utils.data import TensorDataset
        from torch.utils.data.distributed import DistributedSampler

        from keras.src.trainers.data_adapters.torch_data_loader_adapter import (
            TorchDataLoaderAdapter,
        )

        dataset = TensorDataset(
            torch.randn(world_size * 4, 8), torch.randn(world_size * 4, 2)
        )
        dataloader = DataLoader(dataset, batch_size=2)
        with distribution_lib.DataParallel(auto_shard_dataset=False).scope():
            new_dataloader = TorchDataLoaderAdapter(
                dataloader
            ).get_torch_dataloader()
            self.assertNotIsInstance(new_dataloader.sampler, DistributedSampler)

    def test_dataloader_distributed_sampler(self):
        self.run_distributed(
            TorchDataLoadingTest._dataloader_distributed_sampler_test
        )

    def test_dataloader_model_parallel_sampler(self):
        self.run_distributed(
            TorchDataLoadingTest._dataloader_model_parallel_sampler_test
        )

    def test_dataloader_no_auto_shard(self):
        self.run_distributed(
            TorchDataLoadingTest._dataloader_no_auto_shard_test
        )

    def _dataloader_adapter_various_test(self, rank, world_size):
        from torch.utils.data import DataLoader
        from torch.utils.data import TensorDataset

        from keras.src.trainers.data_adapters.torch_data_loader_adapter import (
            TorchDataLoaderAdapter,
        )

        dataset = TensorDataset(torch.randn(11, 8), torch.randn(11, 2))
        dataloader = DataLoader(dataset, batch_size=2)

        # 1. ModelParallel with num_model_replicas >= num_process (1 >= 1)
        mesh = distribution_lib.DeviceMesh((world_size,), ["batch"])
        dist = distribution_lib.ModelParallel(
            layout_map=distribution_lib.LayoutMap(mesh), batch_dim_name="batch"
        )
        with dist.scope():
            adapter = TorchDataLoaderAdapter(dataloader)
            self.assertEqual(adapter.batch_size, 2)
            self.assertEqual(adapter.num_batches, 6)
            self.assertTrue(adapter.has_partial_batch)
            self.assertEqual(adapter.partial_batch_size, 1)

            # Test iterators
            it = adapter.get_numpy_iterator()
            batch = next(it)
            self.assertIsInstance(batch[0], np.ndarray)

            it_jax = adapter.get_jax_iterator()
            batch_jax = next(it_jax)
            self.assertIsInstance(batch_jax[0], np.ndarray)

    def test_dataloader_adapter_various(self):
        self.run_distributed(
            TorchDataLoadingTest._dataloader_adapter_various_test,
            world_size=1,
        )

    def _dataloader_adapter_multi_replica_test(self, rank, world_size):
        from torch.utils.data import DataLoader
        from torch.utils.data import TensorDataset
        from torch.utils.data.distributed import DistributedSampler

        from keras.src.trainers.data_adapters.torch_data_loader_adapter import (
            TorchDataLoaderAdapter,
        )

        dataset = TensorDataset(torch.randn(10, 8), torch.randn(10, 2))
        dataloader = DataLoader(dataset, batch_size=2)

        # mesh size 4, world size 4. batch dim size 2 => 2 processes per replica
        mesh = distribution_lib.DeviceMesh((2, 2), ["batch", "model"])
        dist = distribution_lib.ModelParallel(
            layout_map=distribution_lib.LayoutMap(mesh), batch_dim_name="batch"
        )
        with dist.scope():
            adapter = TorchDataLoaderAdapter(dataloader)
            new_loader = adapter.get_torch_dataloader()
            self.assertIsInstance(new_loader.sampler, DistributedSampler)
            self.assertEqual(new_loader.sampler.num_replicas, 2)
            self.assertEqual(new_loader.sampler.rank, rank // 2)

    def test_dataloader_adapter_multi_replica(self):
        self.run_distributed(
            TorchDataLoadingTest._dataloader_adapter_multi_replica_test,
            world_size=4,
        )

    def test_dataloader_adapter_tf(self):
        from torch.utils.data import DataLoader
        from torch.utils.data import TensorDataset

        from keras.src.trainers.data_adapters.torch_data_loader_adapter import (
            TorchDataLoaderAdapter,
        )

        dataset = TensorDataset(torch.randn(4, 8), torch.randn(4, 2))
        dataloader = DataLoader(dataset, batch_size=2)
        adapter = TorchDataLoaderAdapter(dataloader)
        # Cover get_tf_dataset
        ds = adapter.get_tf_dataset()
        self.assertIsNotNone(ds)

    def test_array_data_adapter_extra(self):
        from keras.src.trainers.data_adapters.array_data_adapter import (
            ArrayDataAdapter,
        )

        x = np.random.randn(10, 8)
        y = np.random.randn(10, 2)

        # 1. Test shuffle="batch"
        adapter = ArrayDataAdapter(x, y, batch_size=2, shuffle="batch")
        it = adapter.get_numpy_iterator()
        next(it)

        # 2. Test shuffle=True (full shuffle)
        adapter = ArrayDataAdapter(x, y, batch_size=2, shuffle=True)
        it = adapter.get_numpy_iterator()
        next(it)

        # 3. Test class_weight error with nested y
        with self.assertRaisesRegex(
            ValueError, "`class_weight` is only supported"
        ):
            ArrayDataAdapter(x, [y, y], class_weight={0: 1.0})

        # 4. Test sample_weight replication
        adapter = ArrayDataAdapter(x, [y, y], sample_weight=np.ones((10, 1)))
        self.assertLen(adapter._inputs[2], 2)

    def test_array_data_adapter_torch_sharding(self):
        from keras.src.trainers.data_adapters.array_data_adapter import (
            ArrayDataAdapter,
        )

        x = np.random.randn(10, 8)
        y = np.random.randn(10, 2)

        # Test shuffle="batch" in torch dataloader
        adapter = ArrayDataAdapter(x, y, batch_size=2, shuffle="batch")
        loader = adapter.get_torch_dataloader()
        self.assertIsNotNone(loader)
        next(iter(loader))

        # Test shuffle=True in torch dataloader
        adapter = ArrayDataAdapter(x, y, batch_size=2, shuffle=True)
        loader = adapter.get_torch_dataloader()
        next(iter(loader))

    def test_dataloader_model_parallel_complex_sharding(self):
        self.run_distributed(
            TorchDataLoadingTest._dataloader_model_parallel_complex_sharding_test,
            world_size=4,
        )


class TorchMetricAggregationTest(TorchDistributedTestCase):
    @staticmethod
    def _sync_metrics_test(self, rank, world_size):
        from keras.src import metrics

        m = metrics.MeanSquaredError()
        mesh = distribution_lib.DeviceMesh((world_size,), ["batch"])
        with distribution_lib.DataParallel(device_mesh=mesh).scope():
            model = models.Sequential([layers.Dense(1, input_shape=(1,))])
            model.compile(metrics=[m])
            model.compute_metrics(
                torch.zeros(1, 1), torch.ones(1, 1), torch.zeros(1, 1)
            )
            m.update_state(
                torch.ones(1, 1), torch.tensor([[1.0 + np.sqrt(rank)]])
            )
            model._sync_metrics()
        self.assertAllClose(m.variables[0].value, 3.0)
        self.assertAllClose(m.variables[1].value, 4.0)

    def test_sync_metrics(self):
        self.run_distributed(TorchMetricAggregationTest._sync_metrics_test)


class TorchCheckpointTest(TorchDistributedTestCase):
    @staticmethod
    def _checkpoint_test(self, rank, world_size):
        import tempfile

        mesh = distribution_lib.DeviceMesh((1, world_size), ["batch", "model"])
        layout_map = distribution_lib.LayoutMap(mesh)
        layout_map[".*dense.*kernel"] = distribution_lib.TensorLayout(
            [None, "model"]
        )
        dist = distribution_lib.ModelParallel(
            layout_map=layout_map, batch_dim_name="model"
        )

        with dist.scope():
            model = models.Sequential(
                [
                    layers.Input(shape=(8,)),
                    layers.Dense(world_size * 4, name="dense_1"),
                ]
            )
            weights = [
                np.ones((8, world_size * 4)),
                np.zeros((world_size * 4,)),
            ]
            model.set_weights(weights)

            with tempfile.NamedTemporaryFile(suffix=".weights.h5") as f:
                model.save_weights(f.name)

                model2 = models.Sequential(
                    [
                        layers.Input(shape=(8,)),
                        layers.Dense(world_size * 4, name="dense_1"),
                    ]
                )
                model2.load_weights(f.name)

                for w1, w2 in zip(model.get_weights(), model2.get_weights()):
                    self.assertAllClose(w1, w2)

    @staticmethod
    def _full_model_checkpoint_test(self, rank, world_size):
        import tempfile

        mesh = distribution_lib.DeviceMesh((1, world_size), ["batch", "model"])
        layout_map = distribution_lib.LayoutMap(mesh)
        layout_map[".*dense.*kernel"] = distribution_lib.TensorLayout(
            [None, "model"]
        )
        dist = distribution_lib.ModelParallel(
            layout_map=layout_map, batch_dim_name="model"
        )

        with dist.scope():
            model = models.Sequential(
                [
                    layers.Input(shape=(8,)),
                    layers.Dense(world_size * 4, name="dense_1"),
                    layers.Dense(2, name="dense_2"),
                ]
            )
            model.compile(optimizer="adam", loss="mse")
            model.train_on_batch(np.random.randn(2, 8), np.random.randn(2, 2))

            with tempfile.NamedTemporaryFile(suffix=".keras") as f:
                model.save(f.name)
                from keras.src.saving import saving_api

                model2 = saving_api.load_model(f.name)
                for w1, w2 in zip(model.get_weights(), model2.get_weights()):
                    self.assertAllClose(w1, w2)
                self.assertGreater(len(model2.optimizer.variables), 0)

    @staticmethod
    def _dp_checkpoint_test(self, rank, world_size):
        import tempfile

        mesh = distribution_lib.DeviceMesh((world_size,), ["batch"])
        dist = distribution_lib.DataParallel(mesh)

        with dist.scope():
            model = models.Sequential(
                [layers.Input(shape=(8,)), layers.Dense(4)]
            )
            model.set_weights([np.ones((8, 4)), np.zeros((4,))])

            with tempfile.NamedTemporaryFile(suffix=".weights.h5") as f:
                model.save_weights(f.name)

                model2 = models.Sequential(
                    [layers.Input(shape=(8,)), layers.Dense(4)]
                )
                model2.load_weights(f.name)

                for w1, w2 in zip(model.get_weights(), model2.get_weights()):
                    self.assertAllClose(w1, w2)

    def test_checkpoint(self):
        self.run_distributed(TorchCheckpointTest._checkpoint_test)

    def test_full_model_checkpoint(self):
        self.run_distributed(TorchCheckpointTest._full_model_checkpoint_test)

    def test_dp_checkpoint(self):
        self.run_distributed(TorchCheckpointTest._dp_checkpoint_test)


class TorchUnbindTest(TorchDistributedTestCase):
    @staticmethod
    def _unbind_test(self, rank, world_size):
        from torch.distributed.tensor import DTensor
        from torch.distributed.tensor import Replicate
        from torch.distributed.tensor import Shard

        mesh = distribution_lib.DeviceMesh((world_size,), ["model"])
        layout = distribution_lib.TensorLayout(["model", None], mesh)
        dist = distribution_lib.ModelParallel(
            device_mesh=mesh, layout_map=distribution_lib.LayoutMap(mesh)
        )

        with dist.scope():
            tensor = torch.randn(world_size * 2, 4)
            dtensor = backend_dlib.distribute_tensor(tensor, layout)

            # 1. Unbind on sharded dimension (0)
            unbound = dtensor.unbind(0)
            self.assertEqual(len(unbound), world_size * 2)
            for t in unbound:
                self.assertIsInstance(t, DTensor)
                self.assertTrue(
                    all(isinstance(p, Replicate) for p in t.placements)
                )

            # 2. Unbind on non-sharded dimension (1)
            unbound = dtensor.unbind(1)
            self.assertEqual(len(unbound), 4)
            for t in unbound:
                self.assertIsInstance(t, DTensor)
                self.assertTrue(
                    any(
                        isinstance(p, Shard) and p.dim == 0
                        for p in t.placements
                    )
                )

            # 3. Unbind with negative dimension (-1)
            unbound = dtensor.unbind(-1)
            self.assertEqual(len(unbound), 4)
            for t in unbound:
                self.assertIsInstance(t, DTensor)

    def test_unbind(self):
        self.run_distributed(TorchUnbindTest._unbind_test)

    @staticmethod
    def _unbind_op_strategy_complex_test(self, rank, world_size):
        from torch.distributed.tensor import Replicate
        from torch.distributed.tensor import Shard

        mesh = distribution_lib.DeviceMesh((world_size,), ["model"])
        dist = distribution_lib.ModelParallel(
            layout_map=distribution_lib.LayoutMap(mesh),
            batch_dim_name="model",
        )

        with dist.scope():
            # 1. Unbind on a sharded dimension
            layout_sharded = distribution_lib.TensorLayout(
                ["model", None], mesh
            )
            tensor = torch.randn(world_size, 4)
            dtensor_sharded = backend_dlib.distribute_tensor(
                tensor, layout_sharded
            )

            # This triggers _unbind_op_strategy with is_sharded = True
            unbound_sharded = torch.unbind(dtensor_sharded, dim=0)
            for t in unbound_sharded:
                # Placements should be Replicate() now
                self.assertTrue(
                    all(isinstance(p, Replicate) for p in t.placements)
                )

            # 2. Unbind on a non-sharded dimension where there is sharding
            # elsewhere
            layout_other_sharded = distribution_lib.TensorLayout(
                [None, "model"], mesh
            )
            tensor2 = torch.randn(4, world_size)
            dtensor_other = backend_dlib.distribute_tensor(
                tensor2, layout_other_sharded
            )

            # This triggers _unbind_op_strategy with is_sharded = False
            # but has sharding on another dim (dim 1)
            # unbind on dim 0
            unbound_other = torch.unbind(dtensor_other, dim=0)
            for t in unbound_other:
                # Dim 1 was "model" (sharded), now it should be dim 0
                self.assertTrue(
                    any(
                        isinstance(p, Shard) and p.dim == 0
                        for p in t.placements
                    )
                )

    def test_unbind_op_strategy_complex(self):
        self.run_distributed(TorchUnbindTest._unbind_op_strategy_complex_test)


class TorchVariableDistributionTest(TorchDistributedTestCase):
    @staticmethod
    def _non_mp_variable_test(self, rank, world_size):
        v = backend.Variable(torch.ones(2, 2))
        self.assertFalse(hasattr(v.value, "device_mesh"))
        self.assertNotEqual(v.value.device.type, "cuda")

    def test_non_mp_variable(self):
        self.run_distributed(
            TorchVariableDistributionTest._non_mp_variable_test
        )

    def _distribute_variable_test(self, rank, world_size):
        from torch.distributed.tensor import DTensor

        mesh = distribution_lib.DeviceMesh((world_size,), ["batch"])
        layout = distribution_lib.TensorLayout(["batch", None], mesh)

        dist = distribution_lib.ModelParallel(
            layout_map=distribution_lib.LayoutMap(mesh)
        )
        with dist.scope():
            # 1. Using TensorLayout
            v = backend_dlib.distribute_variable(
                torch.randn(world_size, 2), layout
            )
            self.assertIsInstance(v, torch.nn.Parameter)
            self.assertIsInstance(v.data, DTensor)

            # 2. Using backend-specific layout
            backend_layout = backend_dlib._to_backend_layout(layout)
            v2 = backend_dlib.distribute_variable(
                torch.randn(world_size, 2), backend_layout
            )
            self.assertIsInstance(v2, torch.nn.Parameter)
            self.assertIsInstance(v2.data, DTensor)

    def test_distribute_variable(self):
        self.run_distributed(
            TorchVariableDistributionTest._distribute_variable_test
        )

    def _variable_dtensor_initialization_test(self, rank, world_size):
        from torch.distributed.tensor import DTensor

        mesh = distribution_lib.DeviceMesh((world_size,), ["batch"])
        layout = distribution_lib.TensorLayout(["batch", None], mesh)
        dist = distribution_lib.ModelParallel(
            layout_map=distribution_lib.LayoutMap(mesh)
        )
        with dist.scope():
            dtensor = backend_dlib.distribute_tensor(
                torch.randn(world_size, 2), layout
            )

        # Initialize Variable with DTensor but NO layout scope
        v = backend.Variable(dtensor)
        self.assertNotIsInstance(v.value, DTensor)
        self.assertIsInstance(v.value, torch.Tensor)
        self.assertEqual(v.value.shape, (1, 2))

    def test_variable_dtensor_initialization(self):
        self.run_distributed(
            TorchVariableDistributionTest._variable_dtensor_initialization_test
        )

    def _distribute_variable_extra_test(self, rank, world_size):
        from torch.distributed.tensor import DTensor

        mesh = distribution_lib.DeviceMesh((world_size,), ["model"])
        layout = distribution_lib.TensorLayout(["model", None], mesh)
        dist = distribution_lib.ModelParallel(
            layout_map=distribution_lib.LayoutMap(mesh),
            batch_dim_name="model",
        )

        with dist.scope():
            tensor = torch.randn(world_size * 2, 2)
            var = backend_dlib.distribute_variable(
                tensor, layout, trainable=True
            )
            self.assertIsInstance(var, torch.nn.Parameter)
            self.assertIsInstance(var.data, DTensor)
            self.assertTrue(var.requires_grad)

    def test_distribute_variable_extra(self):
        self.run_distributed(
            TorchVariableDistributionTest._distribute_variable_extra_test
        )


class TorchPyDatasetAdapterTest(TorchDistributedTestCase):
    @staticmethod
    def _py_dataset_sharding_test(self, rank, world_size):
        from keras.src.trainers.data_adapters.py_dataset_adapter import (
            PyDataset,
        )

        class MyPyDataset(PyDataset):
            def __init__(self, data, **kwargs):
                super().__init__(**kwargs)
                self.data = data

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx], self.data[idx]

        data = np.arange(world_size * 4).reshape(-1, 1).astype("float32")
        dataset = MyPyDataset(data)

        mesh = distribution_lib.DeviceMesh((world_size,), ["batch"])
        dist = distribution_lib.DataParallel(mesh)

        from keras.src.trainers.data_adapters.py_dataset_adapter import (
            PyDatasetAdapter,
        )

        adapter = PyDatasetAdapter(dataset, distribution=dist)

        self.assertEqual(adapter.num_batches, 8)

        batches = list(adapter.get_numpy_iterator())
        self.assertEqual(len(batches), 4)
        expected_indices = data[rank::world_size]
        for i, batch in enumerate(batches):
            self.assertAllClose(batch[0], expected_indices[i])

    def test_py_dataset_sharding(self):
        self.run_distributed(
            TorchPyDatasetAdapterTest._py_dataset_sharding_test
        )


class TorchDataLoaderShuffleTest(TorchDistributedTestCase):
    @staticmethod
    def _dataloader_shuffle_test(self, rank, world_size):
        from torch.utils.data import DataLoader
        from torch.utils.data import TensorDataset

        from keras.src.trainers.data_adapters.torch_data_loader_adapter import (
            TorchDataLoaderAdapter,
        )

        dataset = TensorDataset(torch.randn(10, 1))
        loader_shuffle = DataLoader(dataset, batch_size=2, shuffle=True)
        with distribution_lib.DataParallel().scope():
            adapter = TorchDataLoaderAdapter(loader_shuffle)
            self.assertTrue(adapter._dataloader.sampler.shuffle)

        loader_no_shuffle = DataLoader(dataset, batch_size=2, shuffle=False)
        with distribution_lib.DataParallel().scope():
            adapter = TorchDataLoaderAdapter(loader_no_shuffle)
            self.assertFalse(adapter._dataloader.sampler.shuffle)

    def test_dataloader_shuffle(self):
        self.run_distributed(
            TorchDataLoaderShuffleTest._dataloader_shuffle_test
        )


class TorchDistributionUtilsTest(TorchDistributedTestCase):
    @staticmethod
    def _to_backend_device_test(self, rank, world_size):
        from unittest.mock import patch

        device = backend_dlib._to_backend_device(None)
        if torch.cuda.is_available():
            self.assertEqual(device.type, "cuda")
        else:
            self.assertEqual(device.type, "cpu")

        device = backend_dlib._to_backend_device("gpu:0")
        if torch.cuda.is_available():
            self.assertEqual(device.type, "cuda")
            self.assertEqual(device.index, 0)
        else:
            self.assertEqual(device.type, "cpu")

        # Mock CUDA available
        with patch("torch.cuda.is_available", return_value=True):
            with patch.dict(os.environ, {"LOCAL_RANK": "2"}):
                device = backend_dlib._to_backend_device("gpu")
                self.assertEqual(device.type, "cuda")
                self.assertEqual(device.index, 2)

        device = backend_dlib._to_backend_device("cpu")
        self.assertEqual(device.type, "cpu")

    def test_to_backend_device(self):
        self.run_distributed(TorchDistributionUtilsTest._to_backend_device_test)

    @staticmethod
    def _to_backend_device_cpu_logic_test(self, rank, world_size):
        # Test device_name contains "cpu"
        device = backend_dlib._to_backend_device("Tensor on cpu:0")
        self.assertEqual(device.type, "cpu")

    def test_to_backend_device_cpu_logic(self):
        self.run_distributed(
            TorchDistributionUtilsTest._to_backend_device_cpu_logic_test
        )


class TorchDistributionTensorTest(TorchDistributedTestCase):
    @staticmethod
    def _distribute_data_input_dp_test(self, rank, world_size):
        data = torch.randn(2, 4)
        mesh = distribution_lib.DeviceMesh((world_size,), ["batch"])
        dist = distribution_lib.DataParallel(mesh)

        with dist.scope():
            output = backend_dlib.distribute_data_input(data, None, "batch")
            self.assertIs(output, data)

    @staticmethod
    def _convert_to_tensor_sharding_test(self, rank, world_size):
        from torch.distributed.tensor import DTensor

        mesh = distribution_lib.DeviceMesh((world_size,), ["model"])
        layout_map = distribution_lib.LayoutMap(mesh)
        dist = distribution_lib.ModelParallel(
            layout_map=layout_map, batch_dim_name="model"
        )

        with dist.scope():
            t = backend.convert_to_tensor(
                np.ones((world_size * 2, 4), dtype="float32")
            )
            self.assertIsInstance(t, DTensor)
            self.assertEqual(t.shape, (world_size * 2, 4))

    def test_distribute_data_input_dp(self):
        self.run_distributed(
            TorchDistributionTensorTest._distribute_data_input_dp_test
        )

    def test_convert_to_tensor_sharding(self):
        self.run_distributed(
            TorchDistributionTensorTest._convert_to_tensor_sharding_test
        )

    @staticmethod
    def _distribute_data_input_numpy_global_state_test(self, rank, world_size):
        from torch.distributed.tensor import DTensor

        from keras.src.backend.common import global_state

        mesh = distribution_lib.DeviceMesh((world_size,), ["batch"])
        layout = distribution_lib.TensorLayout(["batch", None], mesh)
        dist = distribution_lib.ModelParallel(
            layout_map=distribution_lib.LayoutMap(mesh),
            batch_dim_name="batch",
        )

        with dist.scope():
            data_np = np.random.randn(world_size * 2, 2).astype("float32")
            output = backend_dlib.distribute_data_input(
                data_np, layout, "batch"
            )
            self.assertIsInstance(output, DTensor)
            self.assertEqual(
                global_state.get_global_attribute("distribution"), dist
            )

    def test_distribute_data_input_numpy_global_state(self):
        self.run_distributed(
            TorchDistributionTensorTest._distribute_data_input_numpy_global_state_test
        )


class TorchTrainerDistributionTest(TorchDistributedTestCase):
    @staticmethod
    def _distribute_inputs_test(self, rank, world_size):
        from torch.distributed.tensor import DTensor
        from torch.distributed.tensor import Replicate

        from keras.src.backend.torch.trainer import TorchTrainer

        trainer = TorchTrainer()
        mesh = distribution_lib.DeviceMesh((world_size,), ["batch"])
        dist = distribution_lib.DataParallel(mesh)

        x = torch.randn(4, 8)
        with dist.scope():
            distributed_x = trainer._distribute_inputs(dist, x)
            self.assertIs(distributed_x, x)

        # Test ModelParallel sharding
        mesh_mp = distribution_lib.DeviceMesh(
            (1, world_size), ["batch", "model"]
        )
        dist_mp = distribution_lib.ModelParallel(
            layout_map=distribution_lib.LayoutMap(mesh_mp),
            batch_dim_name="batch",
        )
        with dist_mp.scope():
            # 1. Normal sharded input
            distributed_x_mp = trainer._distribute_inputs(dist_mp, x)
            self.assertIsInstance(distributed_x_mp, DTensor)
            self.assertEqual(distributed_x_mp.shape, (4, 8))

            # 2. Replicated input (e.g. for symbolic build)
            replicated_x_mp = trainer._distribute_inputs(
                dist_mp, x, replicate=True
            )
            self.assertIsInstance(replicated_x_mp, DTensor)
            self.assertTrue(
                all(
                    isinstance(p, Replicate) for p in replicated_x_mp.placements
                )
            )

    @staticmethod
    def _setup_ddp_lazy_test(self, rank, world_size):
        # Must use a real model with weights for DDP
        model = models.Sequential([layers.Input(shape=(8,)), layers.Dense(2)])
        model.build()

        mesh = distribution_lib.DeviceMesh((world_size,), ["batch"])
        dist = distribution_lib.DataParallel(mesh)

        with dist.scope():
            model._setup_ddp()
            self.assertTrue(hasattr(model, "_ddp_model"))
            self.assertTrue(model._in_ddp_context)
            model_ptr = model._ddp_model

            model._setup_ddp()
            self.assertIs(model._ddp_model, model_ptr)
            self.assertTrue(model._in_ddp_context)

    @staticmethod
    def _train_step_distribution_test(self, rank, world_size):
        x = torch.randn(4, 8)
        y = torch.randn(4, 2)
        data = (x, y)

        # 1. Test DataParallel train_step (uses DDP model)
        mesh_dp = distribution_lib.DeviceMesh((world_size,), ["batch"])
        dist_dp = distribution_lib.DataParallel(mesh_dp)
        with dist_dp.scope():
            model = models.Sequential(
                [layers.Input(shape=(8,)), layers.Dense(2)]
            )
            model.compile(optimizer="adam", loss="mse")
            model._setup_ddp()
            # train_step should distribute inputs and use _ddp_model
            logs = model.train_step(data)
            self.assertIn("loss", logs)

        mesh_mp = distribution_lib.DeviceMesh((world_size,), ["model"])
        layout_map = distribution_lib.LayoutMap(mesh_mp)
        layout_map[".*kernel"] = distribution_lib.TensorLayout([None, "model"])
        dist_mp = distribution_lib.ModelParallel(
            layout_map=layout_map, batch_dim_name=None
        )
        with dist_mp.scope():
            model = models.Sequential(
                [layers.Input(shape=(8,)), layers.Dense(2)]
            )
            model.compile(optimizer="adam", loss="mse")
            logs = model.train_step(data)
            self.assertIn("loss", logs)

    def test_distribute_inputs(self):
        self.run_distributed(
            TorchTrainerDistributionTest._distribute_inputs_test
        )

    def test_setup_ddp_lazy(self):
        self.run_distributed(TorchTrainerDistributionTest._setup_ddp_lazy_test)

    def test_train_step_distribution(self):
        self.run_distributed(
            TorchTrainerDistributionTest._train_step_distribution_test
        )


class TorchTrainerPredictTest(TorchDistributedTestCase):
    @staticmethod
    def _predict_mode_test(self, rank, world_size):
        mesh = distribution_lib.DeviceMesh((world_size,), ["batch"])
        dist = distribution_lib.DataParallel(mesh, auto_shard_dataset=False)
        with dist.scope():
            # Use a layer that records the training mode during call
            class ModeRecordingLayer(layers.Layer):
                def __init__(self, **kwargs):
                    super().__init__(**kwargs)
                    self.recorded_mode = None

                def call(self, x, training=None):
                    self.recorded_mode = training
                    return x

            recorder = ModeRecordingLayer()
            model = models.Sequential(
                [layers.Input(shape=(8,)), recorder, layers.Dense(2)]
            )
            model.compile(optimizer="adam", loss="mse")

            x = np.random.randn(4, 8).astype("float32")

            # predict() should set eval mode (training=False)
            model.predict(x, batch_size=2)
            self.assertFalse(recorder.recorded_mode)

            # predict_on_batch should set eval mode
            model.predict_on_batch(x[:2])
            self.assertFalse(recorder.recorded_mode)

            # train_on_batch should set train mode
            model.train_on_batch(x[:2], np.random.randn(2, 2).astype("float32"))
            self.assertTrue(recorder.recorded_mode)

    def test_predict_mode(self):
        self.run_distributed(TorchTrainerPredictTest._predict_mode_test)


class TorchDistributionExtraCoverageTest(TorchDistributedTestCase):
    def test_get_device_count_initialized(self):
        from unittest.mock import patch

        with patch("torch.distributed.is_initialized", return_value=True):
            with patch("torch.distributed.get_world_size", return_value=8):
                self.assertEqual(backend_dlib.get_device_count(), 8)

    def test_list_devices_mocked(self):
        from unittest.mock import patch

        # Case 1: device_type is None (defaults to "gpu")
        with patch("torch.cuda.device_count", return_value=2):
            with patch.dict(os.environ, {}):
                if "WORLD_SIZE" in os.environ:
                    del os.environ["WORLD_SIZE"]
                devices = backend_dlib.list_devices(None)
                self.assertEqual(devices, ["gpu:0", "gpu:1"])

        # Case 2: device_type is "cpu"
        with patch("torch.cuda.device_count", return_value=2):
            with patch.dict(os.environ, {}):
                if "WORLD_SIZE" in os.environ:
                    del os.environ["WORLD_SIZE"]
                devices = backend_dlib.list_devices("cpu")
                self.assertEqual(devices, ["cpu:0", "cpu:1"])

        # Case 3: is_initialized is True
        with patch("torch.distributed.is_initialized", return_value=True):
            with patch("torch.distributed.get_world_size", return_value=4):
                devices = backend_dlib.list_devices("gpu")
                self.assertEqual(devices, ["gpu:0", "gpu:1", "gpu:2", "gpu:3"])

        # Case 4: device_type is "GPU" (test .lower())
        with patch("torch.cuda.device_count", return_value=1):
            with patch.dict(os.environ, {}):
                if "WORLD_SIZE" in os.environ:
                    del os.environ["WORLD_SIZE"]
                devices = backend_dlib.list_devices("GPU")
                self.assertEqual(devices, ["gpu:0"])

    def test_to_backend_device_mocked(self):
        from unittest.mock import patch

        # 1. device_name contains "cpu"
        device = backend_dlib._to_backend_device("cpu:0")
        self.assertEqual(device.type, "cpu")

        # 2. cuda available
        with patch("torch.cuda.is_available", return_value=True):
            with patch.dict(os.environ, {"LOCAL_RANK": "1"}):
                device = backend_dlib._to_backend_device(None)
                self.assertEqual(device.type, "cuda")
                self.assertEqual(device.index, 1)

        # 3. cuda not available
        with patch("torch.cuda.is_available", return_value=False):
            device = backend_dlib._to_backend_device(None)
            self.assertEqual(device.type, "cpu")

    def test_to_backend_mesh_mocked(self):
        from unittest.mock import patch

        mesh = MagicMock(spec=distribution_lib.DeviceMesh)
        mesh.shape = (2,)
        mesh.axis_names = ["data"]

        with patch(
            "keras.src.backend.torch.distribution_lib.init_device_mesh"
        ) as mock_init:
            backend_dlib._to_backend_mesh(mesh)
            mock_init.assert_called_once()

        # Test with DeviceMesh input (now supported via pass-through)
        from torch.distributed.device_mesh import DeviceMesh as TorchDeviceMesh

        mock_torch_mesh = MagicMock(spec=TorchDeviceMesh)
        self.assertIs(
            backend_dlib._to_backend_mesh(mock_torch_mesh), mock_torch_mesh
        )

    def test_to_backend_layout_mocked(self):
        from unittest.mock import patch

        with patch(
            "keras.src.backend.torch.distribution_lib.list_devices",
            return_value=["gpu:0", "gpu:1"],
        ):
            mesh = distribution_lib.DeviceMesh((2,), ["data"])
        layout = distribution_lib.TensorLayout(["data", None], mesh)

        # Mock _to_backend_mesh
        mock_torch_mesh = MagicMock()
        mock_torch_mesh.axis_names = ("data",)
        with patch(
            "keras.src.backend.torch.distribution_lib._to_backend_mesh",
            return_value=mock_torch_mesh,
        ):
            backend_layout = backend_dlib._to_backend_layout(layout)
            self.assertEqual(backend_layout.device_mesh, mock_torch_mesh)
            from torch.distributed.tensor import Replicate
            from torch.distributed.tensor import Shard

            self.assertIsInstance(backend_layout.placements[0], Shard)
            # axes is empty case for axis_names loop
            layout_no_axes = distribution_lib.TensorLayout([], mesh)
            backend_layout_no_axes = backend_dlib._to_backend_layout(
                layout_no_axes
            )
            self.assertIsInstance(
                backend_layout_no_axes.placements[0], Replicate
            )

    def test_distribute_tensor_mocked(self):
        from unittest.mock import MagicMock
        from unittest.mock import patch

        tensor = torch.randn(4)
        # 1. layout is None
        self.assertIs(backend_dlib.distribute_tensor(tensor, None), tensor)

        # 2. Not ModelParallel
        with patch(
            "keras.src.distribution.distribution_lib.distribution",
            return_value=MagicMock(),
        ):
            self.assertIs(
                backend_dlib.distribute_tensor(tensor, MagicMock()), tensor
            )

        # 3. Already DTensor (redistribute)
        mock_dist = MagicMock(spec=distribution_lib.ModelParallel)
        mock_dtensor = MagicMock(spec=backend_dlib.DTensor)
        mock_layout = MagicMock()
        with patch(
            "keras.src.distribution.distribution_lib.distribution",
            return_value=mock_dist,
        ):
            backend_dlib.distribute_tensor(mock_dtensor, mock_layout)
            mock_dtensor.redistribute.assert_called_once()

    def test_distribute_variable_mocked(self):
        from unittest.mock import MagicMock
        from unittest.mock import patch

        with patch(
            "keras.src.backend.torch.distribution_lib.distribute_tensor"
        ) as mock_dist_tensor:
            mock_dist_tensor.return_value = torch.randn(4)
            res = backend_dlib.distribute_variable(torch.randn(4), MagicMock())
            self.assertIsInstance(res, torch.nn.Parameter)

    def test_distribute_data_input_mocked(self):
        from unittest.mock import MagicMock
        from unittest.mock import patch

        # 1. Not ModelParallel
        tensor = torch.randn(4)
        self.assertIs(
            backend_dlib.distribute_data_input(tensor, None, "batch"), tensor
        )

        # 2. Already DTensor
        mock_dist = MagicMock(spec=distribution_lib.ModelParallel)
        mock_dtensor = MagicMock(spec=backend_dlib.DTensor)
        with patch(
            "keras.src.distribution.distribution_lib.distribution",
            return_value=mock_dist,
        ):
            self.assertIs(
                backend_dlib.distribute_data_input(
                    mock_dtensor, MagicMock(), "batch"
                ),
                mock_dtensor,
            )

        # 3. Meta device
        meta_tensor = torch.randn(4, device="meta")
        with patch(
            "keras.src.distribution.distribution_lib.distribution",
            return_value=mock_dist,
        ):
            self.assertIs(
                backend_dlib.distribute_data_input(
                    meta_tensor, MagicMock(), "batch"
                ),
                meta_tensor,
            )

        # 4. Numpy input
        with patch(
            "keras.src.distribution.distribution_lib.distribution",
            return_value=mock_dist,
        ):
            with patch(
                "keras.src.backend.torch.core.convert_to_tensor"
            ) as mock_conv:
                mock_conv.return_value = torch.randn(4)
                with patch(
                    "keras.src.backend.torch.distribution_lib.DTensor.from_local"
                ) as mock_from_local:
                    backend_dlib.distribute_data_input(
                        np.ones(4), MagicMock(), "batch"
                    )
                    mock_from_local.assert_called_once()

    def test_unbind_op_strategy_mocked(self):
        from unittest.mock import MagicMock

        from torch.distributed.tensor import Shard
        from torch.distributed.tensor._dtensor_spec import DTensorSpec

        # Mock DTensorLayout __init__
        mesh = MagicMock()
        placements = (Shard(0),)
        layout = backend_dlib.DTensorLayout(mesh, placements)
        self.assertEqual(layout.device_mesh, mesh)
        self.assertEqual(layout.placements, placements)

        # Mock _unbind_op_strategy
        # 1. Sharded case
        mock_arg_strategy = MagicMock()
        mock_arg_strategy.output_spec = DTensorSpec(
            mesh=mesh, placements=placements
        )
        mock_input_strategy = MagicMock()
        mock_input_strategy.strategies = [mock_arg_strategy]
        mock_input_strategy.shape = torch.Size([4, 4])
        mock_input_strategy.ndim = 2
        mock_input_strategy.mesh = mesh

        op_schema = MagicMock()
        op_schema.args_schema = (mock_input_strategy, 0)
        strategy = backend_dlib._unbind_op_strategy(op_schema)
        self.assertLen(strategy.strategies, 1)

        # 2. Not sharded case (on sharded dim) - Shard(1), dim=0
        mock_arg_strategy2 = MagicMock()
        mock_arg_strategy2.output_spec = DTensorSpec(
            mesh=mesh, placements=(Shard(1),)
        )
        mock_input_strategy2 = MagicMock()
        mock_input_strategy2.strategies = [mock_arg_strategy2]
        mock_input_strategy2.shape = torch.Size([4, 4])
        mock_input_strategy2.ndim = 2
        mock_input_strategy2.mesh = mesh

        op_schema2 = MagicMock()
        op_schema2.args_schema = (mock_input_strategy2, 0)
        strategy2 = backend_dlib._unbind_op_strategy(op_schema2)
        self.assertLen(strategy2.strategies, 1)

        # 3. Negative dim case
        op_schema3 = MagicMock()
        op_schema3.args_schema = (mock_input_strategy2, -1)
        strategy3 = backend_dlib._unbind_op_strategy(op_schema3)
        self.assertLen(strategy3.strategies, 1)

        # 4. Not sharded case - Shard(0), dim=1 (placement.dim < dim)
        mock_arg_strategy4 = MagicMock()
        mock_arg_strategy4.output_spec = DTensorSpec(
            mesh=mesh, placements=(Shard(0),)
        )
        mock_input_strategy4 = MagicMock()
        mock_input_strategy4.strategies = [mock_arg_strategy4]
        mock_input_strategy4.shape = torch.Size([4, 4])
        mock_input_strategy4.ndim = 2
        mock_input_strategy4.mesh = mesh

        op_schema4 = MagicMock()
        op_schema4.args_schema = (mock_input_strategy4, 1)
        strategy4 = backend_dlib._unbind_op_strategy(op_schema4)
        self.assertLen(strategy4.strategies, 1)

    def test_to_backend_layout_none(self):
        self.assertIsNone(backend_dlib._to_backend_layout(None))

    @staticmethod
    def _to_backend_layout_replicate_test(self, rank, world_size):
        mesh = distribution_lib.DeviceMesh((world_size,), ["model"])
        # Layout doesn't use "model" axis => Replicate
        layout = distribution_lib.TensorLayout([None, None], mesh)
        backend_layout = backend_dlib._to_backend_layout(layout)
        from torch.distributed.tensor import Replicate

        self.assertIsInstance(backend_layout.placements[0], Replicate)

    def test_to_backend_layout_replicate(self):
        self.run_distributed(
            TorchDistributionExtraCoverageTest._to_backend_layout_replicate_test
        )

    def test_register_unbind_twice(self):
        # Should return early
        backend_dlib._register_unbind_strategy()
        backend_dlib._register_unbind_strategy()

    def test_distribute_data_input_none_layout(self):
        tensor = torch.randn(2, 2)
        self.assertIs(
            backend_dlib.distribute_data_input(tensor, None, "batch"), tensor
        )

    @staticmethod
    def _distribute_data_input_not_model_parallel_test(self, rank, world_size):
        tensor = torch.randn(2, 2)
        mesh = distribution_lib.DeviceMesh((world_size,), ["batch"])
        dist = distribution_lib.DataParallel(mesh)
        layout = distribution_lib.TensorLayout(["batch", None], mesh)
        with dist.scope():
            self.assertIs(
                backend_dlib.distribute_data_input(tensor, layout, "batch"),
                tensor,
            )

    def test_distribute_data_input_not_model_parallel(self):
        self.run_distributed(
            TorchDistributionExtraCoverageTest._distribute_data_input_not_model_parallel_test
        )

    def test_fit_validation_split(self):
        model = models.Sequential([layers.Dense(2)])
        model.compile(optimizer="adam", loss="mse")
        x = np.random.randn(10, 8).astype("float32")
        y = np.random.randn(10, 2).astype("float32")
        model.fit(x, y, validation_split=0.2, epochs=1)

    def test_predict_multiple_batches(self):
        model = models.Sequential([layers.Dense(2)])
        model.compile(optimizer="adam", loss="mse")
        x = np.random.randn(10, 8).astype("float32")
        # batch_size=2, num_samples=10 => 5 batches
        res = model.predict(x, batch_size=2)
        self.assertEqual(res.shape, (10, 2))

    def test_array_data_adapter_sample_weight_errors(self):
        from keras.src.trainers.data_adapters.array_data_adapter import (
            ArrayDataAdapter,
        )

        x = np.random.randn(10, 8)
        y = np.random.randn(10, 2)
        # sample_weight and class_weight together
        with self.assertRaisesRegex(
            ValueError, "cannot `class_weight` and `sample_weight`"
        ):
            ArrayDataAdapter(
                x, y, sample_weight=np.ones(10), class_weight={0: 1.0}
            )
