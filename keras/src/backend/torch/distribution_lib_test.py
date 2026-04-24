import os
from unittest import mock

import numpy as np
import pytest
import torch
from torch.distributed.tensor import DTensor
from torch.distributed.tensor import Replicate
from torch.distributed.tensor import Shard

from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src import testing
from keras.src.backend.torch import distribution_lib as backend_dlib
from keras.src.distribution import distribution_lib as dist_lib


@pytest.mark.skipif(backend.backend() != "torch", reason="Torch specific.")
class TorchDistributionLibTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                "gloo", rank=0, world_size=1, init_method="tcp://127.0.0.1:0"
            )

    def test_utils_and_init(self):
        with mock.patch("torch.distributed.is_initialized", return_value=False):
            self.assertEqual(backend_dlib.num_processes(), 1)
            self.assertEqual(backend_dlib.process_id(), 0)
            with mock.patch.dict(os.environ, {"WORLD_SIZE": "2"}):
                self.assertEqual(backend_dlib.get_device_count(), 2)
            with (
                mock.patch("torch.cuda.device_count", return_value=0),
                mock.patch.dict(os.environ, {}, clear=True),
            ):
                self.assertEqual(backend_dlib.get_device_count(), 1)
            with mock.patch("torch.cuda.is_available", return_value=False):
                self.assertEqual(
                    backend_dlib._to_backend_device("gpu").type, "cpu"
                )

        self.assertEqual(backend_dlib._to_backend_device("cpu").type, "cpu")
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.set_device") as mset,
            mock.patch("torch.distributed.init_process_group") as minit,
        ):
            backend_dlib.initialize()
            mset.assert_called()
            minit.assert_called_with(backend="nccl")
            with mock.patch.dict(os.environ, {"LOCAL_RANK": "1"}):
                self.assertEqual(
                    backend_dlib._to_backend_device("gpu").index, 1
                )

    def test_ops_and_conversions(self):
        mesh = dist_lib.DeviceMesh((1,), ["data"], ["cpu:0"])
        tm = backend_dlib._to_backend_mesh(mesh)
        self.assertIs(backend_dlib._to_backend_mesh(tm), tm)
        self.assertIsNone(backend_dlib._to_backend_layout(None))
        layout, t = dist_lib.TensorLayout(["data"], mesh), torch.randn(4)
        self.assertIsInstance(
            backend_dlib._to_backend_layout(layout).placements[0], Shard
        )
        self.assertIs(backend_dlib.distribute_tensor(t, None), t)
        self.assertIs(backend_dlib.distribute_data_input(t, None, "b"), t)
        self.assertIs(backend_dlib.distribute_data_input(t, layout, "b"), t)

        with dist_lib.ModelParallel(
            layout_map=dist_lib.LayoutMap(mesh)
        ).scope():
            self.assertIsInstance(
                backend_dlib.distribute_variable(t, layout), torch.nn.Parameter
            )
            dt = backend_dlib.distribute_data_input(t, layout, "b")
            self.assertIs(
                backend_dlib.distribute_data_input(dt, layout, "b"), dt
            )
            self.assertIsInstance(
                backend_dlib.distribute_data_input(
                    np.ones(4, "f"), layout, "b"
                ),
                DTensor,
            )
            mt = torch.randn(4, device="meta")
            self.assertIs(
                backend_dlib.distribute_data_input(mt, layout, "b"), mt
            )

    def test_unbind_strategy(self):
        mesh = dist_lib.DeviceMesh((1,), ["data"], ["cpu:0"])
        with dist_lib.ModelParallel(
            layout_map=dist_lib.LayoutMap(mesh)
        ).scope():
            dt = backend_dlib.distribute_tensor(
                torch.randn(4, 2), dist_lib.TensorLayout([None, None], mesh)
            )
            for st in torch.unbind(dt, 0):
                self.assertIsInstance(st.placements[0], Replicate)

    def test_e2e_building(self):
        mesh = dist_lib.DeviceMesh((1,), ["data"], ["cpu:0"])
        layout_map = dist_lib.LayoutMap(mesh)
        layout_map[".*kernel"] = dist_lib.TensorLayout([None, "data"])
        dist = dist_lib.ModelParallel(
            layout_map=layout_map, batch_dim_name="data"
        )
        with dist.scope():
            dense = layers.Dense(8)
            dense.build((16, 16))
            self.assertIsInstance(dense.kernel.value.data, DTensor)
            dense.kernel.assign(np.zeros((16, 8), "float32"))
            self.assertIsInstance(dense.kernel.value.data, DTensor)
        with dist_lib.DataParallel().scope():
            self.assertTrue(
                models.Sequential([layers.Input((8,)), layers.Dense(4)])
                .weights[0]
                .trainable
            )
