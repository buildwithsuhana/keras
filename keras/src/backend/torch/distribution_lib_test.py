import os
import socket
from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp
from torch.distributed.tensor import DeviceMesh as TorchDeviceMesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor import Replicate
from torch.distributed.tensor import Shard

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
class TorchDistributionLibTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        # Initialize default process group for single-process unit tests
        if not torch.distributed.is_initialized():
            os.environ.update(
                {"MASTER_ADDR": "localhost", "MASTER_PORT": "12361"}
            )
            torch.distributed.init_process_group("gloo", rank=0, world_size=1)

    def test_get_device_count(self):
        # Default
        self.assertEqual(backend_dlib.get_device_count(), 1)
        # Mocked
        with (
            mock.patch("torch.distributed.is_initialized", return_value=True),
            mock.patch("torch.distributed.get_world_size", return_value=2),
        ):
            self.assertEqual(backend_dlib.get_device_count(), 2)
        with mock.patch("torch.distributed.is_initialized", return_value=False):
            with mock.patch.dict(os.environ, {"WORLD_SIZE": "4"}):
                self.assertEqual(backend_dlib.get_device_count(), 4)
            with (
                mock.patch.dict(os.environ, {}, clear=True),
                mock.patch("torch.cuda.device_count", return_value=3),
            ):
                self.assertEqual(backend_dlib.get_device_count(), 3)

    def test_list_devices(self):
        with mock.patch(
            "keras.src.backend.torch.distribution_lib.get_device_count",
            return_value=2,
        ):
            self.assertEqual(
                backend_dlib.list_devices("gpu"), ["gpu:0", "gpu:1"]
            )

    def test_initialize(self):
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.set_device") as mock_set,
            mock.patch("torch.distributed.init_process_group"),
            mock.patch.dict(os.environ, {"LOCAL_RANK": "1"}),
        ):
            backend_dlib.initialize()
            mock_set.assert_called_once_with(1)
        with (
            mock.patch("torch.cuda.is_available", return_value=False),
            mock.patch("torch.distributed.init_process_group") as mock_init,
        ):
            backend_dlib.initialize()
            mock_init.assert_called_once_with(backend="gloo")

    def test_processes(self):
        # Not initialized
        with mock.patch("torch.distributed.is_initialized", return_value=False):
            self.assertEqual(backend_dlib.num_processes(), 1)
            self.assertEqual(backend_dlib.process_id(), 0)
        # Initialized
        with mock.patch("torch.distributed.is_initialized", return_value=True):
            with mock.patch("torch.distributed.get_world_size", return_value=4):
                self.assertEqual(backend_dlib.num_processes(), 4)
            with mock.patch("torch.distributed.get_rank", return_value=2):
                self.assertEqual(backend_dlib.process_id(), 2)

    def test_to_backend_device(self):
        self.assertEqual(backend_dlib._to_backend_device("cpu").type, "cpu")
        self.assertEqual(backend_dlib._to_backend_device(None).type, "cpu")
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch.dict(os.environ, {"LOCAL_RANK": "1"}),
        ):
            self.assertEqual(
                backend_dlib._to_backend_device("gpu").type, "cuda"
            )
        # Fallback to CPU
        with mock.patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(backend_dlib._to_backend_device("gpu").type, "cpu")

    def test_to_backend_mesh(self):
        mock_mesh = MagicMock(spec=TorchDeviceMesh)
        self.assertIs(backend_dlib._to_backend_mesh(mock_mesh), mock_mesh)
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch(
                "keras.src.backend.torch.distribution_lib.init_device_mesh"
            ) as mock_init,
        ):
            mesh = distribution_lib.DeviceMesh((1,), ["data"], ["cpu:0"])
            backend_dlib._to_backend_mesh(mesh)
            mock_init.assert_called_once_with(
                "cuda", mesh_shape=(1,), mesh_dim_names=("data",)
            )

    def test_to_backend_layout(self):
        self.assertIsNone(backend_dlib._to_backend_layout(None))
        keras_mesh = distribution_lib.DeviceMesh((1,), ["data"], ["cpu:0"])
        mock_layout = MagicMock(axes=None, device_mesh=keras_mesh)
        self.assertIsInstance(
            backend_dlib._to_backend_layout(mock_layout).placements[0],
            Replicate,
        )
        # axes NO match
        mock_layout_no_match = MagicMock(axes=["other"], device_mesh=keras_mesh)
        self.assertIsInstance(
            backend_dlib._to_backend_layout(mock_layout_no_match).placements[0],
            Replicate,
        )

    def test_distribute_tensor(self):
        keras_mesh = distribution_lib.DeviceMesh((1,), ["data"], ["cpu:0"])
        layout = distribution_lib.TensorLayout(["data"], keras_mesh)
        tensor = torch.randn(4)
        # No distribution
        self.assertIs(backend_dlib.distribute_tensor(tensor, None), tensor)
        with mock.patch(
            "keras.src.distribution.distribution_lib.distribution",
            return_value=MagicMock(),
        ):
            self.assertIs(
                backend_dlib.distribute_tensor(tensor, layout), tensor
            )
        # ModelParallel
        dist_parallel = distribution_lib.ModelParallel(
            layout_map=distribution_lib.LayoutMap(keras_mesh)
        )
        with dist_parallel.scope():
            # Layout conversion and redistribution
            backend_layout = backend_dlib._to_backend_layout(layout)
            self.assertIsInstance(
                backend_dlib.distribute_tensor(tensor, backend_layout), DTensor
            )
            dt = backend_dlib.distribute_tensor(tensor, layout)
            self.assertIsInstance(dt, DTensor)
            self.assertIsInstance(
                backend_dlib.distribute_tensor(dt, layout), DTensor
            )

    def test_distribute_variable(self):
        keras_mesh = distribution_lib.DeviceMesh((1,), ["data"], ["cpu:0"])
        layout = distribution_lib.TensorLayout(["data"], keras_mesh)
        dist_parallel = distribution_lib.ModelParallel(
            layout_map=distribution_lib.LayoutMap(keras_mesh)
        )
        with dist_parallel.scope():
            var = backend_dlib.distribute_variable(torch.randn(4), layout)
            self.assertIsInstance(var, torch.nn.Parameter)
            self.assertIsInstance(var.data, DTensor)

    def test_distribute_data_input(self):
        keras_mesh = distribution_lib.DeviceMesh((1,), ["data"], ["cpu:0"])
        layout = distribution_lib.TensorLayout(["data"], keras_mesh)
        tensor = torch.randn(4)
        self.assertIs(
            backend_dlib.distribute_data_input(tensor, None, "batch"), tensor
        )
        with mock.patch(
            "keras.src.distribution.distribution_lib.distribution",
            return_value=MagicMock(),
        ):
            self.assertIs(
                backend_dlib.distribute_data_input(tensor, layout, "batch"),
                tensor,
            )
        dist_parallel = distribution_lib.ModelParallel(
            layout_map=distribution_lib.LayoutMap(keras_mesh)
        )
        with dist_parallel.scope():
            self.assertIsInstance(
                backend_dlib.distribute_data_input(
                    np.ones(4, dtype="float32"), layout, "batch"
                ),
                DTensor,
            )
            # Cover DTensor branch
            dt = backend_dlib.distribute_tensor(tensor, layout)
            self.assertIs(
                backend_dlib.distribute_data_input(dt, layout, "batch"), dt
            )
            # Cover non-TensorLayout branch
            backend_layout = backend_dlib._to_backend_layout(layout)
            self.assertIsInstance(
                backend_dlib.distribute_data_input(
                    np.ones(4, dtype="float32"), backend_layout, "batch"
                ),
                DTensor,
            )
            meta_t = torch.randn(4, device="meta")
            self.assertIs(
                backend_dlib.distribute_data_input(meta_t, layout, "batch"),
                meta_t,
            )

    def test_unbind_strategy(self):
        keras_mesh = distribution_lib.DeviceMesh((1,), ["data"], ["cpu:0"])
        dist_parallel = distribution_lib.ModelParallel(
            layout_map=distribution_lib.LayoutMap(keras_mesh)
        )
        with dist_parallel.scope():
            # Sharded case
            dt_sharded = backend_dlib.distribute_tensor(
                torch.randn(4, 2),
                distribution_lib.TensorLayout(["data", None], keras_mesh),
            )
            for t in torch.unbind(dt_sharded, dim=0):
                self.assertIsInstance(t.placements[0], Replicate)
            # Not sharded case (dim 1)
            for t in torch.unbind(dt_sharded, dim=1):
                self.assertIsInstance(t.placements[0], Shard)
            # Case p.dim > dim (Line 220)
            dt_dim1 = backend_dlib.distribute_tensor(
                torch.randn(4, 2),
                distribution_lib.TensorLayout([None, "data"], keras_mesh),
            )
            for t in torch.unbind(dt_dim1, dim=0):
                self.assertIsInstance(t.placements[0], Shard)
                self.assertEqual(t.placements[0].dim, 0)
            # Negative dim
            for t in torch.unbind(dt_sharded, dim=-1):
                self.assertIsInstance(t.placements[0], Shard)
        # Already registered branch
        with (
            mock.patch(
                "keras.src.backend.torch.distribution_lib._UNBIND_REGISTERED",
                True,
            ),
            mock.patch(
                "torch.distributed.tensor._ops.register_op_strategy"
            ) as mock_reg,
        ):
            backend_dlib._register_unbind_strategy()
            mock_reg.assert_not_called()


@pytest.mark.skipif(
    backend.backend() != "torch", reason="Only for Torch backend"
)
class TorchDistributionIntegrationTest(testing.TestCase):
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
            }
        )
        try:
            if not dist.is_initialized():
                dist.init_process_group(
                    "gloo", rank=rank, world_size=world_size
                )
            test_fn(cls("setUp"), rank, world_size)
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()

    def run_distributed(self, test_fn, world_size=None):
        world_size = world_size or self.world_size
        port = find_free_port()
        mp.spawn(
            TorchDistributionIntegrationTest._worker_wrapper,
            args=(world_size, port, test_fn, self.__class__),
            nprocs=world_size,
            join=True,
        )

    @staticmethod
    def _test_e2e_data_parallel(self, rank, world_size):
        distribution = distribution_lib.DataParallel()
        with distribution.scope():
            model = models.Sequential(
                [layers.Input(shape=(8,)), layers.Dense(4)]
            )
            # For vanilla DataParallel, variables are replicated regular Tensors
            for weight in model.weights:
                self.assertTrue(weight.trainable)

    def test_e2e_data_parallel(self):
        self.run_distributed(
            TorchDistributionIntegrationTest._test_e2e_data_parallel
        )

    @staticmethod
    def _test_e2e_model_parallel(self, rank, world_size):
        mesh = distribution_lib.DeviceMesh((world_size,), ["model"])
        layout_map = distribution_lib.LayoutMap(mesh)
        layout_map[".*dense.*kernel"] = distribution_lib.TensorLayout(
            [None, "model"]
        )
        distribution = distribution_lib.ModelParallel(
            layout_map=layout_map, batch_dim_name="model"
        )
        with distribution.scope():
            model = models.Sequential(
                [
                    layers.Input(shape=(8,)),
                    layers.Dense(world_size * 2, name="dense"),
                ]
            )
            # Kernel should be sharded
            kernel_val = model.layers[0].kernel.value.data
            self.assertIsInstance(kernel_val, DTensor)
            self.assertIsInstance(kernel_val.placements[0], Shard)
            self.assertEqual(kernel_val.placements[0].dim, 1)

    def test_e2e_model_parallel(self):
        self.run_distributed(
            TorchDistributionIntegrationTest._test_e2e_model_parallel
        )
