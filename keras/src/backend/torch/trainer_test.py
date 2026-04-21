import os
import socket
from unittest import mock

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp

from keras.src import backend
from keras.src import callbacks
from keras.src import layers
from keras.src import models
from keras.src import optimizers
from keras.src import testing
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
        if hasattr(torch, "set_default_device"):
            torch.set_default_device("cpu")
        try:
            dist.init_process_group("gloo", rank=rank, world_size=world_size)
            test_fn(cls("setUp"), rank, world_size)
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()

    def run_distributed(self, test_fn, world_size=None):
        previous_device = os.environ.get("KERAS_TORCH_DEVICE")
        os.environ["KERAS_TORCH_DEVICE"] = "cpu"
        world_size, port = world_size or self.world_size, find_free_port()
        try:
            mp.spawn(
                TorchDistributedTestCase._worker_wrapper,
                args=(world_size, port, test_fn, self.__class__),
                nprocs=world_size,
                join=True,
            )
        finally:
            if previous_device is None:
                os.environ.pop("KERAS_TORCH_DEVICE", None)
            else:
                os.environ["KERAS_TORCH_DEVICE"] = previous_device


@pytest.mark.skipif(
    backend.backend() != "torch", reason="Only for Torch backend"
)
class TorchTrainerGeneralTest(testing.TestCase):
    def test_api_and_logic_coverage(self):
        # Covers init, fit, evaluate, predict, *_on_batch, and edge cases
        model = models.Sequential(
            [
                layers.Input(shape=(4,)),
                layers.BatchNormalization(),
                layers.Dense(1),
            ]
        )
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        x, y = np.ones((16, 4), "f"), np.ones((16, 1), "f")
        model._initial_epoch = 1
        model.fit(x, y, epochs=2, batch_size=4, validation_split=0.2, verbose=0)
        from keras.src.backend.torch.trainer import TorchEpochIterator

        model._eval_epoch_iterator = TorchEpochIterator(x=x, y=y, batch_size=4)
        model.evaluate(
            x, y, return_dict=True, verbose=0, _use_cached_eval_dataset=True
        )
        model.train_on_batch(
            x[:4], y[:4], class_weight={0: 1.0}, return_dict=True
        )
        model.test_on_batch(x[:4], y[:4], return_dict=True)
        model.predict_on_batch(x[:4])

        # Errors and warnings
        model.steps_per_execution = 2
        with self.assertRaises(ValueError):
            model.make_train_function(force=True)
        with self.assertRaises(ValueError):
            model.make_test_function(force=True)
        with self.assertRaises(ValueError):
            model.make_predict_function(force=True)
        model.steps_per_execution = 1

        # jit_compile warning
        v = torch.__version__
        torch.__version__ = "2.0.0"
        try:
            model.jit_compile = True
            with self.assertWarnsRegex(UserWarning, "upgrade to torch>=2.1.0"):
                model._should_torch_compile()
        finally:
            torch.__version__ = v

        # Stop training
        class StopCallback(callbacks.Callback):
            def on_train_batch_end(self, batch, logs=None):
                self.model.stop_training = True

        model.fit(
            x, y, epochs=2, batch_size=4, callbacks=[StopCallback()], verbose=0
        )

    def test_module_wrapper_and_compile(self):
        from keras.src.backend.torch.trainer import _KerasModuleWrapper

        model = models.Sequential(
            [layers.BatchNormalization(input_shape=(4,)), layers.Dense(1)]
        )
        wrapper = _KerasModuleWrapper(model)
        self.assertEqual(wrapper(torch.ones((4, 4))).shape, (4, 1))

        model.compile(optimizer="adam", loss="mse")
        with (
            mock.patch(
                "torch.compile", side_effect=lambda x: x
            ) as mock_compile,
            mock.patch.object(
                model, "_should_torch_compile", return_value=True
            ),
        ):
            model.make_train_function()
            model.make_test_function()
            model.make_predict_function()
            self.assertEqual(mock_compile.call_count, 3)

    def test_mocked_distribution(self):
        model = models.Sequential([layers.Dense(1, input_shape=(4,))])
        model.compile(
            optimizer=optimizers.Adam(gradient_accumulation_steps=2), loss="mse"
        )
        self.assertEqual(
            model._distribute_inputs(None, "not_a_tensor"), "not_a_tensor"
        )

        mock_dist = mock.MagicMock(spec=distribution_lib.DataParallel)
        mock_dist.get_data_layout.return_value = "layout"
        mock_dist.batch_dim_name = "batch"

        with (
            mock.patch(
                "keras.src.distribution.distribution_lib.distribution",
                return_value=mock_dist,
            ),
            mock.patch(
                "keras.src.backend.torch.distribution_lib.distribute_data_input",
                side_effect=lambda t, l, n: t,
            ),
            mock.patch(
                "torch.nn.parallel.DistributedDataParallel",
                side_effect=lambda m, **kwargs: m,
            ),
            mock.patch.object(model.optimizer, "apply"),
        ):
            model._setup_ddp()
            model._setup_ddp()  # hits hasattr block
            model._ddp_model = mock.MagicMock(return_value=torch.ones((4, 1)))
            mock_loss = mock.MagicMock()
            with (
                mock.patch.object(
                    model, "_compute_loss", return_value=mock_loss
                ),
                mock.patch.object(model._loss_tracker, "update_state"),
            ):
                # Accumulation step
                model.optimizer._iterations.assign(0)
                model.train_step((torch.ones((4, 4)), torch.ones((4, 1))))
                model._ddp_model.no_sync.assert_called()
                # Update step
                model.optimizer._iterations.assign(1)
                model._ddp_model.no_sync.reset_mock()
                model.train_step((torch.ones((4, 4)),))
                model._ddp_model.no_sync.assert_not_called()

    def test_specific_logic_coverage(self):
        # Covers 85-87, 119, 130-134
        # 85-87: ModelParallel replicate path
        model = models.Sequential([layers.Dense(1, input_shape=(4,))])
        mock_mp = mock.MagicMock(spec=distribution_lib.ModelParallel)
        mock_mesh = mock.MagicMock()
        mock_mesh.axis_names = ["model"]
        mock_mp.device_mesh = mock_mesh
        mock_mp.get_data_layout.return_value = "layout"
        with mock.patch(
            "keras.src.backend.torch.distribution_lib.distribute_data_input",
            return_value=torch.ones((4, 4)),
        ):
            model._distribute_inputs(
                mock_mp, np.ones((4, 4), "f"), replicate=True
            )

        # 119: model without training arg in call
        class NoTrainingArgModel(models.Model):
            def __init__(self):
                super().__init__()
                self.d = layers.Dense(1)

            def call(self, x):
                return self.d(x)

        model = NoTrainingArgModel()
        model.compile(optimizer="adam", loss="mse")
        model.train_on_batch(np.ones((4, 4), "f"), np.ones((4, 1), "f"))

        # 130-134: optimizer is None and no trainable weights
        model = models.Sequential([layers.Activation("relu", input_shape=(4,))])
        model.compile(
            loss="mse"
        )  # optimizer=None by default if not provided? No, compile requires it.
        model.optimizer = None
        with self.assertWarnsRegex(
            UserWarning, "does not have any trainable weights"
        ):
            model.train_on_batch(np.ones((4, 4), "f"), np.ones((4, 1), "f"))


@pytest.mark.skipif(
    backend.backend() != "torch", reason="Only for Torch backend"
)
class TorchTrainerTest(TorchDistributedTestCase):
    @staticmethod
    def _test_ddp_fit(self, rank, world_size):
        inputs = layers.Input(shape=(4,))
        model = models.Model(inputs, layers.Dense(1)(layers.Dense(8)(inputs)))
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        dist = distribution_lib.DataParallel(
            devices=[f"cpu:{i}" for i in range(world_size)]
        )
        dist.auto_shard_dataset = False
        with dist.scope():
            model.make_train_function(force=True)
            self.assertTrue(model._in_ddp_context)
            x_val, y_val = (
                np.random.randn(16, 4).astype("float32"),
                np.random.randn(16, 1).astype("float32"),
            )
            model.fit(x_val, y_val, epochs=1, batch_size=4, verbose=0)
            model.evaluate(x_val, y_val, batch_size=4, verbose=0)
            model.predict(x_val, batch_size=4, verbose=0)

    def test_ddp_fit(self):
        self.run_distributed(TorchTrainerTest._test_ddp_fit)

    @staticmethod
    def _test_gradient_accumulation_no_sync(self, rank, world_size):
        model = models.Sequential([layers.Input(shape=(4,)), layers.Dense(1)])
        model.compile(
            optimizer=optimizers.Adam(gradient_accumulation_steps=2), loss="mse"
        )
        dist = distribution_lib.DataParallel(
            devices=[f"cpu:{i}" for i in range(world_size)]
        )
        dist.auto_shard_dataset = False
        with dist.scope():
            model.make_train_function(force=True)
            x_val, y_val = (
                np.random.randn(4, 4).astype("float32"),
                np.random.randn(4, 1).astype("float32"),
            )
            model.train_on_batch(x_val, y_val)
            self.assertEqual(int(model.optimizer.iterations), 0)
            model.train_on_batch(x_val, y_val)
            self.assertEqual(int(model.optimizer.iterations), 1)

    def test_gradient_accumulation_no_sync(self):
        self.run_distributed(
            TorchTrainerTest._test_gradient_accumulation_no_sync
        )

    @staticmethod
    def _test_sync_metrics_with_dtensor(self, rank, world_size):
        from torch.distributed.tensor import DeviceMesh
        from torch.distributed.tensor import Replicate
        from torch.distributed.tensor import distribute_tensor

        model = models.Sequential([layers.Input(shape=(4,)), layers.Dense(1)])
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        mesh = DeviceMesh("cpu", list(range(world_size)))
        for metric in model.metrics:
            for var in metric.variables:
                var.assign(
                    distribute_tensor(
                        torch.tensor(1.0, device="cpu"), mesh, [Replicate()]
                    )
                )
        with distribution_lib.DataParallel(
            devices=[f"cpu:{i}" for i in range(world_size)]
        ).scope():
            model._sync_metrics()
        for metric in model.metrics:
            for var in metric.variables:
                self.assertAllClose(var.value, torch.tensor(2.0))

    def test_sync_metrics_with_dtensor(self):
        self.run_distributed(TorchTrainerTest._test_sync_metrics_with_dtensor)
