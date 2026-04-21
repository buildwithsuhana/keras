import os
import socket

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp

from keras.src import backend
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
        # Force CPU device for the backend
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
        world_size = world_size or self.world_size
        port = find_free_port()
        try:
            mp.spawn(
                TorchDistributedTestCase._worker_wrapper,
                args=(world_size, port, test_fn, self.__class__),
                nprocs=world_size,
                join=True,
            )
        finally:
            if previous_device is None:
                if "KERAS_TORCH_DEVICE" in os.environ:
                    del os.environ["KERAS_TORCH_DEVICE"]
            else:
                os.environ["KERAS_TORCH_DEVICE"] = previous_device


@pytest.mark.skipif(
    backend.backend() != "torch", reason="Only for Torch backend"
)
class TorchTrainerTest(TorchDistributedTestCase):
    @staticmethod
    def _test_ddp_fit(self, rank, world_size):
        # Create a simple model
        inputs = layers.Input(shape=(4,))
        x = layers.Dense(8)(inputs)
        outputs = layers.Dense(1)(x)
        model = models.Model(inputs, outputs)
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])

        # Create DataParallel distribution
        dist = distribution_lib.DataParallel(
            devices=[f"cpu:{i}" for i in range(world_size)]
        )
        # Disable auto_shard to allow numpy arrays in fit()
        dist.auto_shard_dataset = False

        with dist.scope():
            # Trigger DDP setup
            model.make_train_function(force=True)
            self.assertTrue(model._in_ddp_context)
            self.assertTrue(hasattr(model, "_ddp_model"))

            # Test fit
            x_val = np.random.randn(16, 4).astype("float32")
            y_val = np.random.randn(16, 1).astype("float32")
            model.fit(x_val, y_val, epochs=1, batch_size=4, verbose=0)

            # Test evaluate
            model.evaluate(x_val, y_val, batch_size=4, verbose=0)

            # Test predict
            model.predict(x_val, batch_size=4, verbose=0)

    def test_ddp_fit(self):
        self.run_distributed(TorchTrainerTest._test_ddp_fit)

    @staticmethod
    def _test_gradient_accumulation_no_sync(self, rank, world_size):
        inputs = layers.Input(shape=(4,))
        outputs = layers.Dense(1)(inputs)
        model = models.Model(inputs, outputs)
        optimizer = optimizers.Adam(gradient_accumulation_steps=2)
        model.compile(optimizer=optimizer, loss="mse")

        dist = distribution_lib.DataParallel(
            devices=[f"cpu:{i}" for i in range(world_size)]
        )
        dist.auto_shard_dataset = False

        with dist.scope():
            model.make_train_function(force=True)

            x_val = np.random.randn(4, 4).astype("float32")
            y_val = np.random.randn(4, 1).astype("float32")

            # Step 1: Accumulation (iterations=0)
            model.train_on_batch(x_val, y_val)
            self.assertEqual(int(model.optimizer.iterations), 0)
            # Actually, let's check iterations.
            # In BaseOptimizer.apply:
            # if is_update_step:
            #    self._iterations.assign(self._iterations + 1)

            # Step 2: Update (iterations=1)
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

        inputs = layers.Input(shape=(4,))
        outputs = layers.Dense(1)(inputs)
        model = models.Model(inputs, outputs)
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])

        # Setup for DTensor on CPU
        mesh = DeviceMesh("cpu", list(range(world_size)))

        for metric in model.metrics:
            for var in metric.variables:
                local_val = torch.tensor(1.0, device="cpu")
                # Replicate scalar across mesh
                replicated_tensor = distribute_tensor(
                    local_val, mesh, [Replicate()]
                )
                var.assign(replicated_tensor)

        dist = distribution_lib.DataParallel(
            devices=[f"cpu:{i}" for i in range(world_size)]
        )
        with dist.scope():
            model._sync_metrics()

        for metric in model.metrics:
            for var in metric.variables:
                # Summed up (1.0 + 1.0 = 2.0)
                self.assertAllClose(var.value, torch.tensor(2.0))

    def test_sync_metrics_with_dtensor(self):
        self.run_distributed(TorchTrainerTest._test_sync_metrics_with_dtensor)


@pytest.mark.skipif(
    backend.backend() != "torch", reason="Only for Torch backend"
)
class TorchTrainerExtraCoverageTest(testing.TestCase):
    def test_setup_ddp_no_dist(self):
        from unittest.mock import patch

        # Cover line 64->exit
        model = models.Sequential([layers.Dense(2)])
        with patch(
            "keras.src.distribution.distribution_lib.distribution",
            return_value=None,
        ):
            model._setup_ddp()
            self.assertFalse(hasattr(model, "_ddp_model"))

    def test_setup_ddp_model_parallel(self):
        from unittest.mock import patch

        from keras.src.distribution.distribution_lib import DeviceMesh
        from keras.src.distribution.distribution_lib import LayoutMap

        # Cover line 64->exit with dist that is not DataParallel
        from keras.src.distribution.distribution_lib import ModelParallel

        model = models.Sequential([layers.Dense(2)])
        mesh = DeviceMesh((1,), ["data"])
        dist = ModelParallel(device_mesh=mesh, layout_map=LayoutMap(mesh))
        with patch(
            "keras.src.distribution.distribution_lib.distribution",
            return_value=dist,
        ):
            model._setup_ddp()
            self.assertFalse(hasattr(model, "_ddp_model"))

    def test_sync_metrics_no_dist(self):
        # Coverage for _sync_metrics branch where dist is None
        model = models.Sequential([layers.Dense(2)])
        model.compile(optimizer="adam", loss="mse")
        model._sync_metrics()  # Should do nothing

    def test_distribute_inputs_non_tensor(self):
        # line 93
        model = models.Sequential([layers.Dense(2)])
        from keras.src.distribution.distribution_lib import DataParallel

        dist = DataParallel()
        res = model._distribute_inputs(dist, "not_a_tensor")
        self.assertEqual(res, "not_a_tensor")

    def test_setup_ddp(self):
        from unittest.mock import MagicMock
        from unittest.mock import patch

        from keras.src.distribution.distribution_lib import DataParallel

        model = models.Sequential([layers.Dense(2)])
        dist = DataParallel()
        with patch(
            "keras.src.distribution.distribution_lib.distribution",
            return_value=dist,
        ):
            with patch(
                "torch.nn.parallel.DistributedDataParallel",
                return_value=MagicMock(),
            ) as mock_ddp:
                model._setup_ddp()
                self.assertTrue(model._in_ddp_context)
                self.assertTrue(hasattr(model, "_ddp_model"))
                mock_ddp.assert_called()

    def test_train_step_dataparallel(self):
        from unittest.mock import MagicMock
        from unittest.mock import patch

        from keras.src.distribution.distribution_lib import DataParallel

        model = models.Sequential([layers.Dense(2, input_shape=(8,))])
        model.compile(optimizer="adam", loss="mse")

        dist = DataParallel()
        with patch(
            "keras.src.distribution.distribution_lib.distribution",
            return_value=dist,
        ):
            model._ddp_model = MagicMock()
            y_pred = torch.ones((4, 2), requires_grad=True)
            model._ddp_model.return_value = y_pred
            model._in_ddp_context = True

            mock_loss = torch.tensor(0.5, requires_grad=True)
            with patch.object(model, "_compute_loss", return_value=mock_loss):
                with patch.object(model.optimizer, "apply"):
                    # Cover y is not None (lines 106-111)
                    x = torch.ones((4, 8))
                    y = torch.ones((4, 2))
                    logs = model.train_step((x, y, None))
                    self.assertIn("loss", logs)

                    # Cover y is None
                    model.train_step((x, None, None))

    def test_train_step_gradient_accumulation(self):
        from unittest.mock import MagicMock
        from unittest.mock import patch

        # Cover lines 138-150
        from keras.src.distribution.distribution_lib import DataParallel

        model = models.Sequential([layers.Dense(2, input_shape=(8,))])
        model.compile(optimizer="adam", loss="mse")
        model.optimizer.gradient_accumulation_steps = 2
        model.optimizer._iterations.assign(0)  # Step 1

        dist = DataParallel()
        with patch(
            "keras.src.distribution.distribution_lib.distribution",
            return_value=dist,
        ):
            model._ddp_model = MagicMock()
            model._ddp_model.no_sync.return_value = MagicMock()
            model._in_ddp_context = True

            mock_loss = torch.tensor(0.5, requires_grad=True)
            with patch.object(model, "_compute_loss", return_value=mock_loss):
                with patch.object(model.optimizer, "apply"):
                    # Step 1: iteration 0, (0+1)%2 != 0 -> no_sync
                    x = torch.ones((4, 8))
                    y = torch.ones((4, 2))
                    model.train_step((x, y, None))
                    model._ddp_model.no_sync.assert_called()

                    # Step 2: iteration 1, (1+1)%2 == 0 -> nullcontext
                    model.optimizer._iterations.assign(1)
                    model.train_step((x, y, None))

    def test_test_step_dataparallel(self):
        from unittest.mock import MagicMock
        from unittest.mock import patch

        from keras.src.distribution.distribution_lib import DataParallel

        model = models.Sequential([layers.Dense(2, input_shape=(8,))])
        model.compile(optimizer="adam", loss="mse")

        dist = DataParallel()
        with patch(
            "keras.src.distribution.distribution_lib.distribution",
            return_value=dist,
        ):
            model._ddp_model = MagicMock()
            model._ddp_model.return_value = torch.ones((4, 2))
            model._in_ddp_context = True

            # Cover y is not None (lines 181-184)
            x = torch.ones((4, 8))
            y = torch.ones((4, 2))
            logs = model.test_step((x, y, None))
            self.assertIn("loss", logs)
            model._ddp_model.assert_called()

    def test_test_step_model_parallel(self):
        from unittest.mock import patch

        from keras.src.distribution.distribution_lib import DeviceMesh
        from keras.src.distribution.distribution_lib import LayoutMap

        # Cover lines 181-184 with ModelParallel
        from keras.src.distribution.distribution_lib import ModelParallel

        model = models.Sequential([layers.Dense(2, input_shape=(8,))])
        model.compile(optimizer="adam", loss="mse")

        mesh = DeviceMesh((1,), ["data"])
        dist = ModelParallel(device_mesh=mesh, layout_map=LayoutMap(mesh))

        with patch(
            "keras.src.distribution.distribution_lib.distribution",
            return_value=dist,
        ):
            x = torch.ones((4, 8))
            y = torch.ones((4, 2))
            with patch.object(
                model, "_distribute_inputs", side_effect=lambda d, t: t
            ) as mock_dist_inputs:
                model.test_step((x, y, None))
                self.assertGreaterEqual(mock_dist_inputs.call_count, 2)

    def test_predict_step_dataparallel(self):
        from unittest.mock import MagicMock
        from unittest.mock import patch

        from keras.src.distribution.distribution_lib import DataParallel

        model = models.Sequential([layers.Dense(2, input_shape=(8,))])
        model.compile(optimizer="adam", loss="mse")

        dist = DataParallel()
        with patch(
            "keras.src.distribution.distribution_lib.distribution",
            return_value=dist,
        ):
            model._ddp_model = MagicMock()
            model._ddp_model.return_value = torch.ones((4, 2))
            model._in_ddp_context = True

            x = torch.ones((4, 8))
            res = model.predict_step((x, None, None))
            self.assertEqual(res.shape, (4, 2))
            model._ddp_model.assert_called()

    def test_sync_metrics_regular_tensor(self):
        from unittest.mock import MagicMock
        from unittest.mock import PropertyMock
        from unittest.mock import patch

        # Cover lines 233-235 (skip to_local if not DTensor)
        from keras.src.distribution.distribution_lib import DataParallel
        from keras.src.metrics.reduction_metrics import Mean

        model = models.Sequential([layers.Dense(2)])
        model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])

        dist = DataParallel()
        mock_v = MagicMock()
        mock_v.value = torch.ones((1,))  # Regular tensor

        with patch.object(
            Mean, "variables", new_callable=PropertyMock
        ) as mock_variables:
            mock_variables.return_value = [mock_v]
            with patch(
                "keras.src.distribution.distribution_lib.distribution",
                return_value=dist,
            ):
                with patch(
                    "torch.distributed.is_initialized", return_value=True
                ):
                    with patch(
                        "torch.distributed.all_reduce"
                    ) as mock_all_reduce:
                        model._sync_metrics()
                        mock_all_reduce.assert_called()

    def test_test_step_y_is_none(self):
        # Cover lines 181-184 branch y is None
        from unittest.mock import MagicMock
        from unittest.mock import patch

        from keras.src.distribution.distribution_lib import DataParallel

        model = models.Sequential([layers.Dense(2, input_shape=(8,))])
        model.compile(optimizer="adam", loss="mse")
        dist = DataParallel()
        with patch(
            "keras.src.distribution.distribution_lib.distribution",
            return_value=dist,
        ):
            with patch.object(
                model, "_distribute_inputs", side_effect=lambda d, t: t
            ):
                model._in_ddp_context = True
                model._ddp_model = MagicMock()
                model._ddp_model.return_value = torch.ones((4, 2))
                with patch.object(
                    model, "_compute_loss", return_value=torch.tensor(0.5)
                ):
                    x = torch.ones((4, 8))
                    model.test_step((x, None, None))
                model._ddp_model.assert_called()

    def test_predict_step_extra(self):
        # Cover branches in predict_step
        from unittest.mock import MagicMock
        from unittest.mock import patch

        from keras.src.distribution.distribution_lib import DataParallel

        model = models.Sequential([layers.Dense(2, input_shape=(8,))])
        model.compile(optimizer="adam", loss="mse")

        x = torch.ones((4, 8))

        # Test DDP branch in predict_step
        dist = DataParallel()
        with patch(
            "keras.src.distribution.distribution_lib.distribution",
            return_value=dist,
        ):
            with patch.object(
                model, "_distribute_inputs", side_effect=lambda d, t: t
            ):
                model._in_ddp_context = True
                model._ddp_model = MagicMock()
                model._ddp_model.return_value = torch.ones((4, 2))
                model.predict_step((x, None, None))
                model._ddp_model.assert_called()

        # Test no training arg branch in predict_step (line 219)
        class SimpleModel(models.Model):
            def __init__(self):
                super().__init__()
                self.dense = layers.Dense(2)

            def call(self, x):
                return self.dense(x)

        model2 = SimpleModel()
        model2.compile()
        model2.predict_step((x, None, None))

    def test_fit_ddp_mode(self):
        # Cover lines 418, 440
        from unittest.mock import MagicMock

        model = models.Sequential([layers.Dense(2, input_shape=(8,))])
        model.compile(optimizer="adam", loss="mse")
        model._in_ddp_context = True
        model._ddp_model = MagicMock()
        model.train_function = MagicMock(return_value={})

        x = np.ones((4, 8))
        y = np.ones((4, 2))
        model.fit(x, y, epochs=1, steps_per_epoch=1)
        model._ddp_model.train.assert_called()
        model._ddp_model.eval.assert_called()

    def test_evaluate_ddp_mode(self):
        # Cover line 539
        from unittest.mock import MagicMock

        model = models.Sequential([layers.Dense(2, input_shape=(8,))])
        model.compile(optimizer="adam", loss="mse")
        model._in_ddp_context = True
        model._ddp_model = MagicMock()
        model.test_function = MagicMock(return_value={})

        x = np.ones((4, 8))
        y = np.ones((4, 2))
        model.evaluate(x, y)
        model._ddp_model.eval.assert_called()

    def test_predict_ddp_mode(self):
        # Cover line 603
        from unittest.mock import MagicMock

        model = models.Sequential([layers.Dense(2, input_shape=(8,))])
        model.compile(optimizer="adam", loss="mse")
        model._in_ddp_context = True
        model._ddp_model = MagicMock()
        model.predict_function = MagicMock(return_value=torch.ones((4, 2)))

        x = np.ones((4, 8))
        model.predict(x)
        model._ddp_model.eval.assert_called()

    def test_train_on_batch_ddp_mode(self):
        # Cover line 651
        from unittest.mock import MagicMock

        model = models.Sequential([layers.Dense(2, input_shape=(8,))])
        model.compile(optimizer="adam", loss="mse")
        model._in_ddp_context = True
        model._ddp_model = MagicMock()
        model.train_function = MagicMock(return_value={})

        x = np.ones((4, 8))
        y = np.ones((4, 2))
        model.train_on_batch(x, y)
        model._ddp_model.train.assert_called()

    def test_test_on_batch_ddp_mode(self):
        # Cover line 678
        from unittest.mock import MagicMock

        model = models.Sequential([layers.Dense(2, input_shape=(8,))])
        model.compile(optimizer="adam", loss="mse")
        model._in_ddp_context = True
        model._ddp_model = MagicMock()
        model.test_function = MagicMock(return_value={})

        x = np.ones((4, 8))
        y = np.ones((4, 2))
        model.test_on_batch(x, y)
        model._ddp_model.eval.assert_called()

    def test_predict_on_batch_ddp_mode(self):
        # Cover line 692
        from unittest.mock import MagicMock

        model = models.Sequential([layers.Dense(2, input_shape=(8,))])
        model.compile(optimizer="adam", loss="mse")
        model._in_ddp_context = True
        model._ddp_model = MagicMock()
        model.predict_function = MagicMock(return_value=torch.ones((4, 2)))

        x = np.ones((4, 8))
        model.predict_on_batch(x)
        model._ddp_model.eval.assert_called()
