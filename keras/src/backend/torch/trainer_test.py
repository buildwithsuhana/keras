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
    with socket.socket() as s:
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
        prev = os.environ.get("KERAS_TORCH_DEVICE")
        os.environ["KERAS_TORCH_DEVICE"] = "cpu"
        world_size, port = world_size or self.world_size, find_free_port()
        try:
            mp.spawn(
                self._worker_wrapper,
                args=(world_size, port, test_fn, self.__class__),
                nprocs=world_size,
                join=True,
            )
        finally:
            if prev is None:
                os.environ.pop("KERAS_TORCH_DEVICE", None)
            else:
                os.environ["KERAS_TORCH_DEVICE"] = prev


@pytest.mark.skipif(
    backend.backend() != "torch", reason="Only for Torch backend"
)
class TorchTrainerGeneralTest(testing.TestCase):
    def test_api_and_logic_coverage(self):
        model = models.Sequential(
            [layers.Input((4,)), layers.BatchNormalization(), layers.Dense(1)]
        )
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        x, y = np.ones((16, 4), "f"), np.ones((16, 1), "f")
        model._initial_epoch = 1
        model.fit(x, y, epochs=2, batch_size=4, validation_split=0.2, verbose=0)
        from keras.src.backend.torch.trainer import TorchEpochIterator

        model._eval_epoch_iterator = TorchEpochIterator(x=x, y=y, batch_size=4)
        model.evaluate(x, y, verbose=0, _use_cached_eval_dataset=True)
        model.train_on_batch(x[:4], y[:4], class_weight={0: 1.0})
        model.test_on_batch(x[:4], y[:4])
        model.predict_on_batch(x[:4])
        model.steps_per_execution = 2
        for fn in [
            model.make_train_function,
            model.make_test_function,
            model.make_predict_function,
        ]:
            with self.assertRaises(ValueError):
                fn(force=True)
        model.steps_per_execution = 1
        v = torch.__version__
        torch.__version__ = "2.0.0"
        try:
            model.jit_compile = True
            with self.assertWarnsRegex(UserWarning, "upgrade to torch>=2.1.0"):
                model._should_torch_compile()
        finally:
            torch.__version__ = v

        class StopCallback(callbacks.Callback):
            def on_train_batch_end(self, batch, logs=None):
                self.model.stop_training = True

        model.fit(
            x, y, epochs=2, batch_size=4, callbacks=[StopCallback()], verbose=0
        )
        mock_mp = mock.MagicMock(spec=distribution_lib.ModelParallel)
        mock_mp.device_mesh = mock.MagicMock(axis_names=["model"])
        mock_mp.get_data_layout.return_value = "layout"
        with mock.patch(
            "keras.src.backend.torch.distribution_lib.distribute_data_input",
            return_value=torch.ones((4, 4)),
        ):
            model._distribute_inputs(
                mock_mp, np.ones((4, 4), "f"), replicate=True
            )

        class M(models.Model):
            def __init__(self):
                super().__init__()
                self.d = layers.Dense(1)

            def call(self, x):
                return self.d(x)

        m = M()
        m.compile("adam", "mse")
        m.train_on_batch(x[:4], y[:4])
        model = models.Sequential([layers.Activation("relu", input_shape=(4,))])
        model.compile(loss="mse")
        model.optimizer = None
        with self.assertWarnsRegex(
            UserWarning, "does not have any trainable weights"
        ):
            model.train_on_batch(x[:4], y[:4])

    def test_module_wrapper_and_compile(self):
        from keras.src.backend.torch.trainer import _KerasModuleWrapper

        model = models.Sequential(
            [layers.BatchNormalization(input_shape=(4,)), layers.Dense(1)]
        )
        self.assertEqual(
            _KerasModuleWrapper(model)(torch.ones((4, 4))).shape, (4, 1)
        )
        model.compile(optimizer="adam", loss="mse")
        with (
            mock.patch("torch.compile", side_effect=lambda x: x) as m_compile,
            mock.patch.object(
                model, "_should_torch_compile", return_value=True
            ),
        ):
            model.make_train_function()
            model.make_test_function()
            model.make_predict_function()
            self.assertEqual(m_compile.call_count, 3)

    def test_mocked_distribution(self):
        model = models.Sequential([layers.Dense(1, input_shape=(4,))])
        model.compile(
            optimizer=optimizers.Adam(gradient_accumulation_steps=2), loss="mse"
        )
        self.assertEqual(model._distribute_inputs(None, "t"), "t")
        mock_dist = mock.MagicMock(spec=distribution_lib.DataParallel)
        mock_dist.get_data_layout.return_value, mock_dist.batch_dim_name = (
            "layout",
            "batch",
        )
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
            model._setup_ddp()
            model._ddp_model = mock.MagicMock(return_value=torch.ones((4, 1)))
            with (
                mock.patch.object(
                    model, "_compute_loss", return_value=mock.MagicMock()
                ),
                mock.patch.object(model._loss_tracker, "update_state"),
            ):
                for i, sync in [(0, True), (1, False)]:
                    model.optimizer._iterations.assign(i)
                    if not sync:
                        model._ddp_model.no_sync.reset_mock()
                    model.train_step((torch.ones((4, 4)), torch.ones((4, 1))))
                    if sync:
                        model._ddp_model.no_sync.assert_called()
                    else:
                        model._ddp_model.no_sync.assert_not_called()


@pytest.mark.skipif(
    backend.backend() != "torch", reason="Only for Torch backend"
)
class TorchTrainerTest(TorchDistributedTestCase):
    @staticmethod
    def _test_distributed_all(self, rank, world_size):
        model = models.Sequential(
            [layers.Dense(8, input_shape=(4,)), layers.Dense(1)]
        )
        model.compile(
            optimizer=optimizers.Adam(gradient_accumulation_steps=2),
            loss="mse",
            metrics=["mae"],
        )
        dist = distribution_lib.DataParallel(
            devices=[f"cpu:{i}" for i in range(world_size)]
        )
        dist.auto_shard_dataset = False
        with dist.scope():
            model.make_train_function(force=True)
            self.assertTrue(model._in_ddp_context)
            x, y = np.ones((16, 4), "f"), np.ones((16, 1), "f")
            model.fit(x, y, epochs=1, batch_size=4, verbose=0)
            model.evaluate(x, y, batch_size=4, verbose=0)
            model.predict(x, batch_size=4, verbose=0)
            # fit did 4 batches, so 2 updates. iterations=2.
            # Next train_on_batch is step 5, no update.
            model.train_on_batch(x[:4], y[:4])
            self.assertEqual(int(model.optimizer.iterations), 2)
            # Next train_on_batch is step 6, update!
            model.train_on_batch(x[:4], y[:4])
            self.assertEqual(int(model.optimizer.iterations), 3)

    def test_distributed_all(self):
        self.run_distributed(self._test_distributed_all)

    @staticmethod
    def _test_sync_metrics_with_dtensor(self, rank, world_size):
        from torch.distributed.tensor import DeviceMesh
        from torch.distributed.tensor import Replicate
        from torch.distributed.tensor import distribute_tensor

        model = models.Sequential([layers.Input((4,)), layers.Dense(1)])
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
        self.run_distributed(self._test_sync_metrics_with_dtensor)
