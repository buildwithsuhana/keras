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


@pytest.mark.skipif(
    backend.backend() != "torch", reason="Only for Torch backend"
)
class TorchDistributionLibTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        if not torch.distributed.is_initialized():
            os.environ.update(
                {"MASTER_ADDR": "localhost", "MASTER_PORT": "12361"}
            )
            torch.distributed.init_process_group("gloo", rank=0, world_size=1)

    def test_backend_info(self):
        # 38, 44: num_processes/process_id when not initialized
        with mock.patch("torch.distributed.is_initialized", return_value=False):
            self.assertEqual(backend_dlib.num_processes(), 1)
            self.assertEqual(backend_dlib.process_id(), 0)
            # 14-16: get_device_count fallback branches
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

    def test_list_devices(self):
        # 21-23: list_devices coverage
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

    def test_backend_conversions(self):
        # 54: _to_backend_device fallback to cpu
        self.assertEqual(backend_dlib._to_backend_device("cpu").type, "cpu")
        with mock.patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(backend_dlib._to_backend_device("gpu").type, "cpu")
        # 53: _to_backend_device with cuda available
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch.dict(os.environ, {"LOCAL_RANK": "1"}),
        ):
            device = backend_dlib._to_backend_device("gpu")
            self.assertEqual(device.type, "cuda")
            self.assertEqual(device.index, 1)

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
            mock_init.assert_called_once()

        self.assertIsNone(backend_dlib._to_backend_layout(None))
        # 90-95: Axis name matching in _to_backend_layout
        layout = distribution_lib.TensorLayout(["data"], mesh)
        backend_layout = backend_dlib._to_backend_layout(layout)
        self.assertIsInstance(backend_layout.placements[0], Shard)
        self.assertEqual(backend_layout.placements[0].dim, 0)

        mock_layout = MagicMock(axes=None, device_mesh=mesh)
        self.assertIsInstance(
            backend_dlib._to_backend_layout(mock_layout).placements[0],
            Replicate,
        )

    def test_distribution_ops(self):
        keras_mesh = distribution_lib.DeviceMesh((1,), ["data"], ["cpu:0"])
        layout = distribution_lib.TensorLayout(["data"], keras_mesh)
        tensor = torch.randn(4)

        # 106, 112, 140: Return tensor if not ModelParallel or layout is None
        self.assertIs(backend_dlib.distribute_tensor(tensor, None), tensor)
        self.assertIs(backend_dlib.distribute_tensor(tensor, layout), tensor)
        self.assertIs(
            backend_dlib.distribute_data_input(tensor, None, "batch"), tensor
        )
        self.assertIs(
            backend_dlib.distribute_data_input(tensor, layout, "batch"), tensor
        )

        dist_p = distribution_lib.ModelParallel(
            layout_map=distribution_lib.LayoutMap(keras_mesh)
        )
        with dist_p.scope():
            # 122: distribute_tensor for regular tensor
            dt = backend_dlib.distribute_tensor(tensor, layout)
            self.assertIsInstance(dt, DTensor)
            # 118-121: redistribute DTensor
            dt2 = backend_dlib.distribute_tensor(dt, layout)
            self.assertIsInstance(dt2, DTensor)

            # 146: Return already DTensor
            self.assertIs(
                backend_dlib.distribute_data_input(dt, layout, "batch"), dt
            )

            # 152-155: Convert non-tensor input while clearing distribution
            self.assertIsInstance(
                backend_dlib.distribute_data_input(
                    np.ones(4, "float32"), layout, "batch"
                ),
                DTensor,
            )

            meta_t = torch.randn(4, device="meta")
            self.assertIs(
                backend_dlib.distribute_data_input(meta_t, layout, "batch"),
                meta_t,
            )

            # 133-134: distribute_variable
            var = backend_dlib.distribute_variable(tensor, layout)
            self.assertIsInstance(var, torch.nn.Parameter)
            self.assertIsInstance(var.data, DTensor)

    def test_unbind_strategy(self):
        keras_mesh = distribution_lib.DeviceMesh((1,), ["data"], ["cpu:0"])
        with distribution_lib.ModelParallel(
            layout_map=distribution_lib.LayoutMap(keras_mesh)
        ).scope():
            # 220: Case p.dim > dim in _unbind_op_strategy
            dt = backend_dlib.distribute_tensor(
                torch.randn(4, 2),
                distribution_lib.TensorLayout([None, "data"], keras_mesh),
            )
            for t in torch.unbind(dt, dim=0):
                self.assertIsInstance(t.placements[0], Shard)
                self.assertEqual(t.placements[0].dim, 0)

            # 201-210, 222: Case is_sharded and Replicate coverage
            dt_sharded = backend_dlib.distribute_tensor(
                torch.randn(4, 2),
                distribution_lib.TensorLayout(["data", None], keras_mesh),
            )
            for t in torch.unbind(dt_sharded, dim=0):
                self.assertIsInstance(t.placements[0], Replicate)

            dt_replicate = backend_dlib.distribute_tensor(
                torch.randn(4, 2),
                distribution_lib.TensorLayout([None, None], keras_mesh),
            )
            for t in torch.unbind(dt_replicate, dim=0):
                self.assertIsInstance(t.placements[0], Replicate)


@pytest.mark.skipif(
    backend.backend() != "torch", reason="Only for Torch backend"
)
class TorchDistributionIntegrationTest(testing.TestCase):
    def run_distributed(self, test_fn):
        port = find_free_port()
        mp.spawn(
            _worker_wrapper,
            args=(2, port, test_fn, self.__class__),
            nprocs=2,
            join=True,
        )

    def test_e2e_data_parallel(self):
        self.run_distributed(_test_e2e_data_parallel_fn)

    def test_e2e_model_parallel(self):
        self.run_distributed(_test_e2e_model_parallel_fn)


def _test_e2e_data_parallel_fn(self, rank, ws):
    with distribution_lib.DataParallel().scope():
        model = models.Sequential([layers.Input(shape=(8,)), layers.Dense(4)])
        for weight in model.weights:
            self.assertTrue(weight.trainable)


def _test_e2e_model_parallel_fn(self, rank, ws):
    mesh = distribution_lib.DeviceMesh((ws,), ["model"])
    layout_map = distribution_lib.LayoutMap(mesh)
    layout_map[".*dense.*kernel"] = distribution_lib.TensorLayout(
        [None, "model"]
    )
    with distribution_lib.ModelParallel(
        layout_map=layout_map, batch_dim_name="model"
    ).scope():
        model = models.Sequential(
            [layers.Input(shape=(8,)), layers.Dense(ws * 2, name="dense")]
        )
        kernel_val = model.layers[0].kernel.value.data
        self.assertIsInstance(kernel_val, DTensor)
        self.assertEqual(kernel_val.placements[0].dim, 1)
