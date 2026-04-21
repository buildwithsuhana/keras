import os
import socket
from unittest import mock

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp
from torch.distributed.tensor import DeviceMesh as TDM
from torch.distributed.tensor import DTensor
from torch.distributed.tensor import Replicate
from torch.distributed.tensor import Shard

from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src import testing
from keras.src.backend.torch import distribution_lib as bdl
from keras.src.distribution import distribution_lib as dl


def find_free_port():
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _worker(rank, ws, port, fn, cls):
    os.environ.update(
        {
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": str(port),
            "RANK": str(rank),
            "WORLD_SIZE": str(ws),
            "LOCAL_RANK": str(rank),
        }
    )
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group("gloo", rank=rank, world_size=ws)
    try:
        fn(cls("setUp"), rank, ws)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def _test_e2e(self, rank, ws):
    with dl.DataParallel().scope():
        for w in models.Sequential(
            [layers.Input((8,)), layers.Dense(4)]
        ).weights:
            self.assertTrue(w.trainable)
    mesh = dl.DeviceMesh((ws,), ["model"])
    lm = dl.LayoutMap(mesh)
    lm[".*dense.*kernel"] = dl.TensorLayout([None, "model"])
    with dl.ModelParallel(layout_map=lm, batch_dim_name="model").scope():
        m = models.Sequential(
            [layers.Input((8,)), layers.Dense(ws * 2, name="dense")]
        )
        k = m.layers[0].kernel.value.data
        self.assertIsInstance(k, DTensor)
        self.assertEqual(k.placements[0].dim, 1)


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
            self.assertEqual(bdl.num_processes(), 1)
            self.assertEqual(bdl.process_id(), 0)
            with mock.patch.dict(os.environ, {"WORLD_SIZE": "4"}):
                self.assertEqual(bdl.get_device_count(), 4)
            with (
                mock.patch.dict(os.environ, {}, clear=True),
                mock.patch("torch.cuda.device_count", return_value=0),
            ):
                self.assertEqual(bdl.get_device_count(), 1)
        with (
            mock.patch("torch.distributed.is_initialized", return_value=True),
            mock.patch("torch.distributed.get_world_size", return_value=2),
            mock.patch("torch.distributed.get_rank", return_value=1),
        ):
            self.assertEqual(bdl.get_device_count(), 2)
            self.assertEqual(bdl.num_processes(), 2)
            self.assertEqual(bdl.process_id(), 1)
        with mock.patch(
            "keras.src.backend.torch.distribution_lib.get_device_count",
            return_value=2,
        ):
            self.assertEqual(bdl.list_devices(), ["gpu:0", "gpu:1"])
            self.assertEqual(bdl.list_devices("cpu"), ["cpu:0", "cpu:1"])

    def test_initialize(self):
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.set_device") as m_set,
            mock.patch("torch.distributed.init_process_group"),
            mock.patch.dict(os.environ, {"LOCAL_RANK": "1"}),
        ):
            bdl.initialize()
            m_set.assert_called_once_with(1)
        with (
            mock.patch("torch.cuda.is_available", return_value=False),
            mock.patch("torch.distributed.init_process_group") as m_init,
        ):
            bdl.initialize()
            m_init.assert_called_once_with(backend="gloo")

    def test_conversions_and_ops(self):
        self.assertEqual(bdl._to_backend_device("cpu").type, "cpu")
        with mock.patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(bdl._to_backend_device("gpu").type, "cpu")
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch.dict(os.environ, {"LOCAL_RANK": "1"}),
        ):
            d = bdl._to_backend_device("gpu")
            self.assertEqual(d.type, "cuda")
            self.assertEqual(d.index, 1)
        m_mesh = mock.MagicMock(spec=TDM)
        self.assertIs(bdl._to_backend_mesh(m_mesh), m_mesh)
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch(
                "keras.src.backend.torch.distribution_lib.init_device_mesh"
            ) as m_init,
        ):
            bdl._to_backend_mesh(dl.DeviceMesh((1,), ["data"], ["cpu:0"]))
            m_init.assert_called_once()
        self.assertIsNone(bdl._to_backend_layout(None))
        mesh = dl.DeviceMesh((1,), ["data"], ["cpu:0"])
        l1 = bdl._to_backend_layout(dl.TensorLayout(["data"], mesh))
        self.assertIsInstance(l1.placements[0], Shard)
        self.assertEqual(l1.placements[0].dim, 0)
        self.assertIsInstance(
            bdl._to_backend_layout(
                mock.MagicMock(axes=None, device_mesh=mesh)
            ).placements[0],
            Replicate,
        )

        layout, tensor = dl.TensorLayout(["data"], mesh), torch.randn(4)
        for t in [None, layout]:
            self.assertIs(bdl.distribute_tensor(tensor, t), tensor)
        for t in [None, layout]:
            self.assertIs(bdl.distribute_data_input(tensor, t, "b"), tensor)
        with dl.ModelParallel(layout_map=dl.LayoutMap(mesh)).scope():
            dt = bdl.distribute_tensor(tensor, layout)
            self.assertIsInstance(dt, DTensor)
            self.assertIsInstance(bdl.distribute_tensor(dt, layout), DTensor)
            self.assertIs(bdl.distribute_data_input(dt, layout, "b"), dt)
            self.assertIsInstance(
                bdl.distribute_data_input(np.ones(4, "f"), layout, "b"), DTensor
            )
            meta = torch.randn(4, device="meta")
            self.assertIs(bdl.distribute_data_input(meta, layout, "b"), meta)
            v = bdl.distribute_variable(tensor, layout)
            self.assertIsInstance(v, torch.nn.Parameter)
            self.assertIsInstance(v.data, DTensor)
            # Unbind
            dt2 = bdl.distribute_tensor(
                torch.randn(4, 2), dl.TensorLayout([None, "data"], mesh)
            )
            for t in torch.unbind(dt2, 0):
                self.assertIsInstance(t.placements[0], Shard)
                self.assertEqual(t.placements[0].dim, 0)
            dt3 = bdl.distribute_tensor(
                torch.randn(4, 2), dl.TensorLayout(["data", None], mesh)
            )
            for t in torch.unbind(dt3, 0):
                self.assertIsInstance(t.placements[0], Replicate)
            dt4 = bdl.distribute_tensor(
                torch.randn(4, 2), dl.TensorLayout([None, None], mesh)
            )
            for t in torch.unbind(dt4, 0):
                self.assertIsInstance(t.placements[0], Replicate)


@pytest.mark.skipif(backend.backend() != "torch", reason="Only for Torch")
class TorchDistributionIntegrationTest(testing.TestCase):
    def test_e2e(self):
        port = find_free_port()
        mp.spawn(
            _worker,
            args=(2, port, _test_e2e, self.__class__),
            nprocs=2,
            join=True,
        )
