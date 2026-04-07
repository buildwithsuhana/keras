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


def distributed_test(world_size=None):
    """Mark a test as distributed.

    Can be used as:
    @distributed_test
    @distributed_test(world_size=4)
    @distributed_test(world_size=[2, 4])
    """

    if callable(world_size):
        # Used as @distributed_test
        fn = world_size
        fn._is_distributed_test = True
        fn._world_size = None
        return fn

    def decorator(fn):
        fn._is_distributed_test = True
        fn._world_size = world_size
        return fn

    return decorator


@pytest.mark.skipif(
    backend.backend() != "torch", reason="Only for Torch backend"
)
class TorchDistributedTestCase(testing.TestCase):
    def setUp(self):
        super().setUp()
        self.world_size = 2

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for name, attr in list(cls.__dict__.items()):
            if not getattr(attr, "_is_distributed_test", False):
                continue
            ws_requested = getattr(attr, "_world_size", None)
            world_sizes = (
                [None]
                if ws_requested is None
                else (
                    list(ws_requested)
                    if isinstance(ws_requested, (list, tuple))
                    else [ws_requested]
                )
            )
            impl_name = f"_impl_{name}"
            setattr(cls, impl_name, attr)

            for ws in world_sizes:
                test_name = name if len(world_sizes) == 1 else f"{name}_ws{ws}"
                wrapper = cls._make_distributed_wrapper(impl_name, ws)
                wrapper.__name__ = test_name
                wrapper.__module__ = attr.__module__
                setattr(cls, test_name, wrapper)

    @staticmethod
    def _make_distributed_wrapper(method_name, world_size):
        def wrapper(self):
            self.run_distributed(method_name, world_size)

        return wrapper

    @staticmethod
    def _setup_distributed_env(rank, world_size):
        """Setup environment variables for distributed training."""
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

    @staticmethod
    def _worker_wrapper(rank, world_size, method_name, cls):
        import torch.distributed as dist

        TorchDistributedTestCase._setup_distributed_env(rank, world_size)
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        backend_dlib.initialize()

        instance = cls("setUp")
        instance.setUp()
        getattr(instance, method_name)(rank, world_size)

        if dist.is_initialized():
            dist.destroy_process_group()

    def run_distributed(self, method_name, world_size=None):
        ws = world_size or self.world_size
        mp.spawn(
            TorchDistributedTestCase._worker_wrapper,
            args=(ws, method_name, self.__class__),
            nprocs=ws,
            join=True,
        )

    # Helper methods for common test patterns
    @staticmethod
    def _create_single_axis_mesh(world_size, axis_name="data"):
        return distribution_lib.DeviceMesh((world_size,), [axis_name])

    @staticmethod
    def _create_layout(axes, mesh):
        return distribution_lib.TensorLayout(axes, mesh)

    @staticmethod
    def _create_model_parallel(layout_map, batch_dim="batch", auto_shard=False):
        return distribution_lib.ModelParallel(
            layout_map=layout_map,
            batch_dim_name=batch_dim,
            auto_shard_dataset=auto_shard,
        )


class TorchDeviceDiscoveryTest(TorchDistributedTestCase):
    @distributed_test
    def test_list_devices(self, rank, world_size):
        devices = distribution_lib.list_devices()
        self.assertLen(devices, world_size)
        self.assertTrue(all(f":{i}" in devices[i] for i in range(world_size)))

    @distributed_test
    def test_get_device_count(self, rank, world_size):
        self.assertEqual(backend_dlib.get_device_count(), world_size)


class TorchProcessManagementTest(TorchDistributedTestCase):
    @distributed_test
    def test_num_processes(self, rank, world_size):
        self.assertEqual(backend_dlib.num_processes(), world_size)

    @distributed_test
    def test_process_id(self, rank, world_size):
        self.assertEqual(backend_dlib.process_id(), rank)


class TorchDeviceMeshMappingTest(TorchDistributedTestCase):
    @distributed_test
    def test_to_backend_mesh(self, rank, world_size):
        from torch.distributed.device_mesh import DeviceMesh

        mesh = self._create_single_axis_mesh(world_size, "data")
        torch_mesh = backend_dlib._to_backend_mesh(mesh)
        self.assertIsInstance(torch_mesh, DeviceMesh)
        self.assertEqual(torch_mesh.mesh.shape, (world_size,))
        self.assertEqual(torch_mesh.mesh_dim_names, ("data",))

    @distributed_test
    def test_to_backend_layout(self, rank, world_size):
        from torch.distributed.tensor import Shard

        mesh = self._create_single_axis_mesh(world_size, "data")
        layout = self._create_layout(["data", None], mesh)
        torch_layout = backend_dlib._to_backend_layout(layout)
        placements = torch_layout.placements
        self.assertLen(placements, 1)
        self.assertIsInstance(placements[0], Shard)
        self.assertEqual(placements[0].dim, 0)


class TorchTensorDistributionTest(TorchDistributedTestCase):
    @distributed_test
    def test_distribute_tensor(self, rank, world_size):
        from torch.distributed.tensor import DTensor

        mesh = self._create_single_axis_mesh(world_size, "data")
        layout = self._create_layout(["data", None], mesh)
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

    @distributed_test
    def test_distribute_data_input(self, rank, world_size):
        from torch.distributed.tensor import DTensor

        mesh = self._create_single_axis_mesh(world_size, "batch")
        layout = self._create_layout(["batch", None], mesh)
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


class TorchVariableDistributionAwarenessTest(TorchDistributedTestCase):
    @distributed_test
    def test_variable_distribution_awareness(self, rank, world_size):
        from torch.distributed.tensor import DTensor

        mesh = self._create_single_axis_mesh(world_size, "model")
        layout_map = distribution_lib.LayoutMap(mesh)
        layout_map[".*dense.*kernel"] = self._create_layout(
            [None, "model"], mesh
        )
        dist = self._create_model_parallel(layout_map, batch_dim="batch")
        with dist.scope():
            layer = layers.Dense(world_size * 4)
            layer.build((8, 8))
        self.assertIsInstance(layer.kernel.value, DTensor)
        self.assertEqual(layer.kernel.value.to_local().shape, (8, 4))


class TorchTrainerArchitectureTest(TorchDistributedTestCase):
    @distributed_test
    def test_e2e_data_parallel_fit(self, rank, world_size):
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

    @distributed_test
    def test_e2e_model_parallel_fit(self, rank, world_size):
        mesh = distribution_lib.DeviceMesh((1, world_size), ["batch", "model"])
        layout_map = distribution_lib.LayoutMap(mesh)
        layout_map[".*dense.*kernel"] = self._create_layout(
            [None, "model"], mesh
        )
        dist = self._create_model_parallel(
            layout_map, batch_dim="batch", auto_shard=0
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

    @distributed_test(world_size=1)
    def test_keras_module_wrapper(self, rank, world_size):
        from keras.src.backend.torch.trainer import _KerasModuleWrapper

        model = models.Sequential(
            [layers.Input(shape=(8,)), layers.Dense(4), layers.Dense(2)]
        )
        model.build()
        wrapper = _KerasModuleWrapper(model)
        self.assertLen(list(wrapper.parameters()), 4)
        self.assertEqual(wrapper(torch.randn(2, 8)).shape, (2, 2))


class TorchDataLoadingTest(TorchDistributedTestCase):
    @distributed_test
    def test_dataloader_distributed_sampler(self, rank, world_size):
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


class TorchMetricAggregationTest(TorchDistributedTestCase):
    @distributed_test
    def test_sync_metrics(self, rank, world_size):
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
