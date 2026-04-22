import contextlib
from unittest import mock

import numpy as np
import pytest
import torch
from torch.distributed.tensor import DeviceMesh
from torch.distributed.tensor import Replicate
from torch.distributed.tensor import distribute_tensor

from keras.src import backend
from keras.src import callbacks
from keras.src import layers
from keras.src import models
from keras.src import optimizers
from keras.src import testing
from keras.src.backend.torch.trainer import TorchEpochIterator
from keras.src.backend.torch.trainer import _KerasModuleWrapper
from keras.src.distribution import distribution_lib


@pytest.mark.skipif(
    backend.backend() != "torch", reason="Only for Torch backend"
)
class TorchTrainerTest(testing.TestCase):
    def test_api_and_logic_coverage(self):
        model = models.Sequential(
            [layers.Input((4,)), layers.BatchNormalization(), layers.Dense(1)]
        )
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        inputs, targets = np.ones((16, 4), "f"), np.ones((16, 1), "f")
        model.fit(
            inputs,
            targets,
            epochs=1,
            batch_size=4,
            validation_split=0.2,
            verbose=0,
        )
        model._eval_epoch_iterator = TorchEpochIterator(
            x=inputs, y=targets, batch_size=4
        )
        model.evaluate(
            inputs, targets, verbose=0, _use_cached_eval_dataset=True
        )
        for batch_fn in [model.train_on_batch, model.test_on_batch]:
            batch_fn(inputs[:4], targets[:4])
        model.predict_on_batch(inputs[:4])
        model.steps_per_execution = 2
        self.assertRaises(ValueError, model.make_train_function, force=True)
        model.steps_per_execution = 1
        model.stop_training = False

        class StopCallback(callbacks.Callback):
            def on_batch_end(self, batch, logs=None):
                self.model.stop_training = True

        model.fit(
            inputs,
            targets,
            epochs=1,
            batch_size=4,
            callbacks=[StopCallback()],
            verbose=0,
        )
        distribution = distribution_lib.ModelParallel(
            layout_map=distribution_lib.LayoutMap(
                distribution_lib.DeviceMesh((1,), ["model"], ["cpu"])
            )
        )

        with mock.patch(
            "keras.src.backend.torch.distribution_lib.distribute_data_input",
            return_value=torch.ones((4, 4)),
        ):
            model._distribute_inputs(
                distribution, np.ones((4, 4), "f"), replicate=True
            )
        small_model = models.Sequential([layers.Dense(1, input_shape=(4,))])
        small_model.compile("adam", "mse")
        small_model.train_on_batch(inputs[:4], targets[:4])
        no_weights_model = models.Sequential(
            [layers.Activation("relu", input_shape=(4,))]
        )
        no_weights_model.compile(loss="mse")
        no_weights_model.optimizer = None
        with self.assertWarnsRegex(
            UserWarning, "does not have any trainable weights"
        ):
            no_weights_model.train_on_batch(inputs[:4], targets[:4])

    def test_module_wrapper_and_compile(self):
        model = models.Sequential(
            [layers.BatchNormalization(input_shape=(4,)), layers.Dense(1)]
        )
        self.assertEqual(
            _KerasModuleWrapper(model)(torch.ones((4, 4))).shape, (4, 1)
        )
        model.compile(optimizer="adam", loss="mse")

        with (
            mock.patch(
                "torch.compile", side_effect=lambda x, **kwargs: x
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
        with (
            mock.patch("torch.distributed.is_initialized", return_value=True),
            mock.patch("torch.distributed.get_world_size", return_value=1),
            mock.patch("torch.distributed.get_rank", return_value=0),
        ):
            distribution = distribution_lib.DataParallel(devices=["cpu:0"])

        with (
            mock.patch(
                "keras.src.distribution.distribution_lib.distribution",
                return_value=distribution,
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
            model._ddp_model = mock.Mock(
                side_effect=lambda *args, **kwargs: torch.randn(
                    (4, 1), requires_grad=True
                )
            )
            model._ddp_model.no_sync.return_value = contextlib.nullcontext()
            for i, expect_no_sync in [(0, True), (1, False)]:
                model.optimizer._iterations.assign(i)
                model.train_step((torch.ones((4, 4)), torch.ones((4, 1))))
                self.assertEqual(
                    model._ddp_model.no_sync.called, expect_no_sync
                )
                model._ddp_model.no_sync.reset_mock()

    def test_distributed_logic(self):
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                "gloo", init_method="tcp://127.0.0.1:0", rank=0, world_size=1
            )
            self.addCleanup(torch.distributed.destroy_process_group)
        model = models.Sequential(
            [layers.Dense(8, input_shape=(4,)), layers.Dense(1)]
        )
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])

        with mock.patch("torch.distributed.all_reduce"):
            distribution = distribution_lib.DataParallel(devices=["cpu:0"])
            distribution.auto_shard_dataset = False
            with distribution.scope():
                model.make_train_function(force=True)
                inputs, targets = np.ones((16, 4), "f"), np.ones((16, 1), "f")
                model.fit(inputs, targets, epochs=1, batch_size=4, verbose=0)
                model.evaluate(inputs, targets, verbose=0)
                model.predict(inputs, verbose=0)
            mesh = DeviceMesh("cpu", [0])
            for var in [v for m in model.metrics for v in m.variables]:
                var.assign(
                    distribute_tensor(torch.tensor(1.0), mesh, [Replicate()])
                )
            with mock.patch(
                "torch.distributed.all_reduce",
                side_effect=lambda t, **kwargs: t.add_(1.0),
            ):
                with distribution.scope():
                    model._sync_metrics()
            for var in [v for m in model.metrics for v in m.variables]:
                self.assertAllClose(var.value, torch.tensor(2.0))
