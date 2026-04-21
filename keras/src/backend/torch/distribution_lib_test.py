import os
import socket
from unittest import mock

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
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _worker_wrapper(rank, world_size, port, test_fn, cls):
    os.environ.update(
        {
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": str(port),
            "RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "LOCAL_RANK": str(rank),
        }
    )
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            "gloo", rank=rank, world_size=world_size
        )
    try:
        test_fn(cls("setUp"), rank, world_size)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def _test_e2e_distributed(self, rank, world_size):
    with distribution_lib.DataParallel().scope():
        model = models.Sequential([layers.Input((8,)), layers.Dense(4)])
        for weight in model.weights:
            self.assertTrue(weight.trainable)
    mesh = distribution_lib.DeviceMesh((world_size,), ["model"])
    layout_map = distribution_lib.LayoutMap(mesh)
    layout_map[".*dense.*kernel"] = distribution_lib.TensorLayout(
        [None, "model"]
    )
    with distribution_lib.ModelParallel(
        layout_map=layout_map, batch_dim_name="model"
    ).scope():
        model = models.Sequential(
            [layers.Input((8,)), layers.Dense(world_size * 2, name="dense")]
        )
        kernel_val = model.layers[0].kernel.value.data
        self.assertIsInstance(kernel_val, DTensor)
        self.assertEqual(kernel_val.placements[0].dim, 1)


@pytest.mark.skipif(backend.backend() != "torch", reason="Only for Torch")
class TorchDistributionLibTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        if not torch.distributed.is_initialized():
            os.environ.update(
                {"MASTER_ADDR": "localhost", "MASTER_PORT": "12362"}
            )
            torch.distributed.init_process_group("gloo", rank=0, world_size=1)

    def test_backend_and_devices(self):
        with mock.patch("torch.distributed.is_initialized", return_value=False):
            self.assertEqual(backend_dlib.num_processes(), 1)
            self.assertEqual(backend_dlib.process_id(), 0)
            with mock.patch.dict(os.environ, {"WORLD_SIZE": "4"}):
                self.assertEqual(backend_dlib.get_device_count(), 4)
            with (
                mock.patch.dict(os.environ, {}, clear=True),
                mock.patch("torch.cuda.device_count", return_value=0),
            ):
                self.assertEqual(backend_dlib.get_device_count(), 1)
        with (
            mock.patch("torch.distributed.is_initialized", return_value=True),
            mock.patch("torch.distributed.get_world_size", return_value=2),
            mock.patch("torch.distributed.get_rank", return_value=1),
        ):
            self.assertEqual(backend_dlib.get_device_count(), 2)
            self.assertEqual(backend_dlib.num_processes(), 2)
            self.assertEqual(backend_dlib.process_id(), 1)
        with mock.patch(
            "keras.src.backend.torch.distribution_lib.get_device_count",
            return_value=2,
        ):
            self.assertEqual(backend_dlib.list_devices(), ["gpu:0", "gpu:1"])
            self.assertEqual(
                backend_dlib.list_devices("cpu"), ["cpu:0", "cpu:1"]
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

    def test_conversions_and_ops(self):
        self.assertEqual(backend_dlib._to_backend_device("cpu").type, "cpu")
        with mock.patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(backend_dlib._to_backend_device("gpu").type, "cpu")
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch.dict(os.environ, {"LOCAL_RANK": "1"}),
        ):
            device = backend_dlib._to_backend_device("gpu")
            self.assertEqual(device.type, "cuda")
            self.assertEqual(device.index, 1)

        mock_mesh = mock.MagicMock(spec=TorchDeviceMesh)
        self.assertIs(backend_dlib._to_backend_mesh(mock_mesh), mock_mesh)
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch(
                "keras.src.backend.torch.distribution_lib.init_device_mesh"
            ) as mock_init,
        ):
            backend_dlib._to_backend_mesh(
                distribution_lib.DeviceMesh((1,), ["data"], ["cpu:0"])
            )
            mock_init.assert_called_once()

        self.assertIsNone(backend_dlib._to_backend_layout(None))
        mesh = distribution_lib.DeviceMesh((1,), ["data"], ["cpu:0"])
        layout = backend_dlib._to_backend_layout(
            distribution_lib.TensorLayout(["data"], mesh)
        )
        self.assertIsInstance(layout.placements[0], Shard)
        self.assertEqual(layout.placements[0].dim, 0)
        self.assertIsInstance(
            backend_dlib._to_backend_layout(
                mock.MagicMock(axes=None, device_mesh=mesh)
            ).placements[0],
            Replicate,
        )

        layout = distribution_lib.TensorLayout(["data"], mesh)
        tensor = torch.randn(4)
        for t in [None, layout]:
            self.assertIs(backend_dlib.distribute_tensor(tensor, t), tensor)
        for t in [None, layout]:
            self.assertIs(
                backend_dlib.distribute_data_input(tensor, t, "b"), tensor
            )

        with distribution_lib.ModelParallel(
            layout_map=distribution_lib.LayoutMap(mesh)
        ).scope():
            dt = backend_dlib.distribute_tensor(tensor, layout)
            self.assertIsInstance(dt, DTensor)
            self.assertIsInstance(
                backend_dlib.distribute_tensor(dt, layout), DTensor
            )
            self.assertIs(
                backend_dlib.distribute_data_input(dt, layout, "b"), dt
            )
            self.assertIsInstance(
                backend_dlib.distribute_data_input(
                    np.ones(4, "f"), layout, "b"
                ),
                DTensor,
            )
            meta = torch.randn(4, device="meta")
            self.assertIs(
                backend_dlib.distribute_data_input(meta, layout, "b"), meta
            )
            v = backend_dlib.distribute_variable(tensor, layout)
            self.assertIsInstance(v, torch.nn.Parameter)
            self.assertIsInstance(v.data, DTensor)

            # Unbind
            dt2 = backend_dlib.distribute_tensor(
                torch.randn(4, 2),
                distribution_lib.TensorLayout([None, "data"], mesh),
            )
            for t in torch.unbind(dt2, 0):
                self.assertIsInstance(t.placements[0], Shard)
                self.assertEqual(t.placements[0].dim, 0)
            dt3 = backend_dlib.distribute_tensor(
                torch.randn(4, 2),
                distribution_lib.TensorLayout(["data", None], mesh),
            )
            for t in torch.unbind(dt3, 0):
                self.assertIsInstance(t.placements[0], Replicate)
            dt4 = backend_dlib.distribute_tensor(
                torch.randn(4, 2),
                distribution_lib.TensorLayout([None, None], mesh),
            )
            for t in torch.unbind(dt4, 0):
                self.assertIsInstance(t.placements[0], Replicate)


@pytest.mark.skipif(backend.backend() != "torch", reason="Only for Torch")
class TorchDistributionIntegrationTest(testing.TestCase):
    def test_e2e(self):
        port = find_free_port()
        mp.spawn(
            _worker_wrapper,
            args=(2, port, _test_e2e_distributed, self.__class__),
            nprocs=2,
            join=True,
        )
