import numpy as np
import pytest
import torch
from absl.testing import parameterized

from keras.src import backend
from keras.src import layers
from keras.src import losses
from keras.src import metrics
from keras.src import models
from keras.src import optimizers
from keras.src import testing
from keras.src.backend.torch import distribution_lib
from keras.src.distribution.distribution_lib import DataParallel
from keras.src.distribution.distribution_lib import DeviceMesh
from keras.src.distribution.distribution_lib import ModelParallel
from keras.src.distribution.distribution_lib import set_distribution


class TorchTrainerTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    def tearDown(self):
        super().tearDown()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    def _ensure_distributed_initialized(self):
        if not torch.distributed.is_initialized():
            import os

            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "29500"
            distribution_lib.initialize(num_processes=1, process_id=0)

    @pytest.mark.skipif(backend.backend() != "torch", reason="Requires torch")
    def test_train_on_batch_with_data_parallel(self):
        self._ensure_distributed_initialized()
        mesh = DeviceMesh(
            shape=(1,), axis_names=["batch"], devices=["cpu:0"]
        )
        dist = DataParallel(device_mesh=mesh)
        set_distribution(dist)

        model = models.Sequential(
            [
                layers.Dense(2, input_shape=(3,)),
            ]
        )
        model.compile(
            optimizer=optimizers.SGD(0.01),
            loss=losses.MeanSquaredError(),
            metrics=[metrics.MeanSquaredError()],
        )

        x = np.ones((4, 3)).astype("float32")
        y = np.ones((4, 2)).astype("float32")

        model.train_on_batch(x, y)
        set_distribution(None)

    @pytest.mark.skipif(backend.backend() != "torch", reason="Requires torch")
    def test_train_on_batch_with_model_parallel(self):
        self._ensure_distributed_initialized()
        mesh = DeviceMesh(
            shape=(1,), axis_names=["model"], devices=["cpu:0"]
        )
        layout_map = ModelParallel.generate_layout_map(
            models.Sequential([layers.Dense(2, input_shape=(3,))]),
            mesh
        )
        dist = ModelParallel(device_mesh=mesh, layout_map=layout_map)
        set_distribution(dist)

        model = models.Sequential(
            [
                layers.Dense(2, input_shape=(3,)),
            ]
        )
        model.compile(
            optimizer=optimizers.SGD(0.01),
            loss=losses.MeanSquaredError(),
            metrics=[metrics.MeanSquaredError()],
        )

        x = np.ones((4, 3)).astype("float32")
        y = np.ones((4, 2)).astype("float32")

        model.train_on_batch(x, y)
        set_distribution(None)

    @pytest.mark.skipif(backend.backend() != "torch", reason="Requires torch")
    def test_fit_with_data_parallel(self):
        self._ensure_distributed_initialized()
        mesh = DeviceMesh(
            shape=(1,), axis_names=["batch"], devices=["cpu:0"]
        )
        dist = DataParallel(device_mesh=mesh)
        set_distribution(dist)

        model = models.Sequential(
            [
                layers.Dense(2, input_shape=(3,)),
            ]
        )
        model.compile(
            optimizer=optimizers.SGD(0.01),
            loss=losses.MeanSquaredError(),
            metrics=[metrics.MeanSquaredError()],
        )

        x = np.ones((4, 3)).astype("float32")
        y = np.ones((4, 2)).astype("float32")

        model.fit(x, y, epochs=1)
        set_distribution(None)
