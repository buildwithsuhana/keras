import os

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


@pytest.mark.skipif(
    backend.backend() != "torch", reason="Only for Torch backend"
)
class TorchDistributedTestCase(testing.TestCase):
    def setUp(self):
        super().setUp()
        self.world_size = 2

    @staticmethod
    def _worker_wrapper(rank, world_size, test_fn, cls):
        import torch.distributed as dist

        os.environ.update(
            {
                "MASTER_ADDR": "localhost",
                "MASTER_PORT": os.environ.get("MASTER_PORT", "29500"),
                "RANK": str(rank),
                "WORLD_SIZE": str(world_size),
                "LOCAL_RANK": str(rank),
                "KERAS_TORCH_DEVICE": "cpu",
                "PYTORCH_ENABLE_MPS_FALLBACK": "1",
            }
        )
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        test_fn(cls("setUp"), rank, world_size)
        if dist.is_initialized():
            dist.destroy_process_group()

    def run_distributed(self, test_fn, world_size=None):
        mp.spawn(
            TorchDistributedTestCase._worker_wrapper,
            args=(world_size or self.world_size, test_fn, self.__class__),
            nprocs=world_size or self.world_size,
            join=True,
        )


class TorchDeviceDiscoveryTest(TorchDistributedTestCase):
    @staticmethod
    def _list_devices_test(self, rank, world_size):
        devices = distribution_lib.list_devices()
        self.assertLen(devices, world_size)
        self.assertTrue(all(f":{i}" in devices[i] for i in range(world_size)))

    @staticmethod
    def _get_device_count_test(self, rank, world_size):
        self.assertEqual(backend_dlib.get_device_count(), world_size)

    def test_list_devices(self):
        self.run_distributed(TorchDeviceDiscoveryTest._list_devices_test)

    def test_get_device_count(self):
        self.run_distributed(TorchDeviceDiscoveryTest._get_device_count_test)


class TorchProcessManagementTest(TorchDistributedTestCase):
    @staticmethod
    def _num_processes_test(self, rank, world_size):
        self.assertEqual(backend_dlib.num_processes(), world_size)

    @staticmethod
    def _process_id_test(self, rank, world_size):
        self.assertEqual(backend_dlib.process_id(), rank)

    def test_num_processes(self):
        self.run_distributed(TorchProcessManagementTest._num_processes_test)

    def test_process_id(self):
        self.run_distributed(TorchProcessManagementTest._process_id_test)


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


class TorchVariableDistributionAwarenessTest(TorchDistributedTestCase):
    @staticmethod
    def _variable_distribution_awareness_test(self, rank, world_size):
        from torch.distributed.tensor import DTensor

        mesh = distribution_lib.DeviceMesh((world_size,), ["model"])
        layout_map = distribution_lib.LayoutMap(mesh)
        layout_map[".*dense.*kernel"] = distribution_lib.TensorLayout(
            [None, "model"]
        )
        dist = distribution_lib.ModelParallel(
            layout_map=layout_map, batch_dim_name="batch"
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

        model = models.Sequential(
            [layers.Input(shape=(8,)), layers.Dense(4), layers.Dense(2)]
        )
        model.build()
        wrapper = _KerasModuleWrapper(model)
        self.assertLen(list(wrapper.parameters()), 4)
        self.assertEqual(wrapper(torch.randn(2, 8)).shape, (2, 2))

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

    def test_dataloader_distributed_sampler(self):
        self.run_distributed(
            TorchDataLoadingTest._dataloader_distributed_sampler_test
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
            layout_map=layout_map, batch_dim_name="batch"
        )

        with dist.scope():
            model = models.Sequential(
                [
                    layers.Input(shape=(8,)),
                    layers.Dense(world_size * 4, name="dense_1"),
                ]
            )
            # Set some predictable weights
            weights = [
                np.ones((8, world_size * 4)),
                np.zeros((world_size * 4,)),
            ]
            model.set_weights(weights)

            with tempfile.NamedTemporaryFile(suffix=".weights.h5") as f:
                model.save_weights(f.name)

                # Create a new model and load weights
                model2 = models.Sequential(
                    [
                        layers.Input(shape=(8,)),
                        layers.Dense(world_size * 4, name="dense_1"),
                    ]
                )
                model2.load_weights(f.name)

                for w1, w2 in zip(model.get_weights(), model2.get_weights()):
                    self.assertAllClose(w1, w2)

    def test_checkpoint(self):
        self.run_distributed(TorchCheckpointTest._checkpoint_test)


class TorchUnbindTest(TorchDistributedTestCase):
    @staticmethod
    def _unbind_test(self, rank, world_size):
        from torch.distributed.tensor import DTensor
        from torch.distributed.tensor import Replicate

        mesh = distribution_lib.DeviceMesh((world_size,), ["model"])
        layout = distribution_lib.TensorLayout(["model", None], mesh)
        dist = distribution_lib.ModelParallel(
            device_mesh=mesh, layout_map=distribution_lib.LayoutMap(mesh)
        )

        with dist.scope():
            tensor = torch.randn(world_size * 2, 4)
            dtensor = backend_dlib.distribute_tensor(tensor, layout)

            # Test unbind along sharded dimension
            unbound = dtensor.unbind(0)
            self.assertEqual(len(unbound), world_size * 2)
            for t in unbound:
                self.assertIsInstance(t, DTensor)
                # Should be replicated now
                self.assertTrue(
                    all(isinstance(p, Replicate) for p in t.placements)
                )

    def test_unbind(self):
        self.run_distributed(TorchUnbindTest._unbind_test)


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

        # num_batches reflects the global dataset
        self.assertEqual(adapter.num_batches, 8)

        batches = list(adapter.get_numpy_iterator())
        # The sharded iterator should only yield batches for this rank
        self.assertEqual(len(batches), 4)

        # Verify indices - each worker gets 4 batches out of 8 total
        # Total data size is world_size * 4 = 8.
        # Rank 0 gets indices 0, 2, 4, 6
        # Rank 1 gets indices 1, 3, 5, 7
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
        # Test with shuffle=True
        loader_shuffle = DataLoader(dataset, batch_size=2, shuffle=True)
        with distribution_lib.DataParallel().scope():
            adapter = TorchDataLoaderAdapter(loader_shuffle)
            self.assertTrue(adapter._dataloader.sampler.shuffle)

        # Test with shuffle=False
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
        # Test None input
        device = backend_dlib._to_backend_device(None)
        self.assertEqual(device.type, "cpu")

        # Test explicit gpu input
        device = backend_dlib._to_backend_device("gpu:0")
        self.assertEqual(device.type, "cuda")
        self.assertEqual(device.index, 0)

        # Test explicit cpu input
        device = backend_dlib._to_backend_device("cpu")
        self.assertEqual(device.type, "cpu")

    def test_to_backend_device(self):
        self.run_distributed(TorchDistributionUtilsTest._to_backend_device_test)


class TorchDistributionTensorTest(TorchDistributedTestCase):
    @staticmethod
    def _distribute_data_input_dp_test(self, rank, world_size):
        # DataParallel distribute_data_input should be a no-op
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
            layout_map=layout_map, batch_dim_name="batch"
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


class TorchTrainerDistributionTest(TorchDistributedTestCase):
    @staticmethod
    def _distribute_inputs_test(self, rank, world_size):
        from torch.distributed.tensor import DTensor

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
            distributed_x_mp = trainer._distribute_inputs(dist_mp, x)
            self.assertIsInstance(distributed_x_mp, DTensor)
            self.assertEqual(distributed_x_mp.shape, (4, 8))

    @staticmethod
    def _setup_ddp_lazy_test(self, rank, world_size):
        # Must use a real model with weights for DDP
        model = models.Sequential([layers.Input(shape=(8,)), layers.Dense(2)])
        model.build()

        mesh = distribution_lib.DeviceMesh((world_size,), ["batch"])
        dist = distribution_lib.DataParallel(mesh)

        with dist.scope():
            # _setup_ddp should be lazy and idempotent
            model._setup_ddp()
            self.assertTrue(hasattr(model, "_ddp_model"))
            model_ptr = model._ddp_model

            model._setup_ddp()
            self.assertIs(model._ddp_model, model_ptr)

    def test_distribute_inputs(self):
        self.run_distributed(
            TorchTrainerDistributionTest._distribute_inputs_test
        )

    def test_setup_ddp_lazy(self):
        self.run_distributed(TorchTrainerDistributionTest._setup_ddp_lazy_test)
