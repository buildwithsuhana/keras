import contextlib
import os
from unittest import mock

import numpy as np
import pytest
import torch

from keras.src import backend
from keras.src import callbacks
from keras.src import layers
from keras.src import models
from keras.src import optimizers
from keras.src import testing
from keras.src.backend.torch.trainer import TorchEpochIterator
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
            with distribution.scope():
                model._distribute_data(np.ones((4, 4), "f"), replicate=True)
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

    def test_model_parallel_training(self):
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                "gloo", init_method="tcp://127.0.0.1:0", rank=0, world_size=1
            )
            self.addCleanup(torch.distributed.destroy_process_group)

        mesh = distribution_lib.DeviceMesh((1,), ["data"], ["cpu"])
        layout_map = distribution_lib.LayoutMap(mesh)
        layout_map[".*kernel"] = distribution_lib.TensorLayout(
            [None, "data"], mesh
        )
        distribution = distribution_lib.ModelParallel(
            layout_map=layout_map, batch_dim_name="data"
        )

        with distribution.scope():
            model = models.Sequential(
                [layers.Dense(8, input_shape=(4,)), layers.Dense(1)]
            )
            model.compile(optimizer="adam", loss="mse")
            inputs, targets = np.ones((16, 4), "f"), np.ones((16, 1), "f")
            model.fit(inputs, targets, epochs=1, batch_size=4, verbose=0)
            model.evaluate(inputs, targets, verbose=0)
            model.predict(inputs, verbose=0)

    def test_distributed_checkpointing(self):
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                "gloo", init_method="tcp://127.0.0.1:0", rank=0, world_size=1
            )
            self.addCleanup(torch.distributed.destroy_process_group)

        model = models.Sequential(
            [layers.Dense(8, input_shape=(4,)), layers.Dense(1)]
        )
        model.compile(optimizer="adam", loss="mse")
        inputs, targets = np.ones((16, 4), "f"), np.ones((16, 1), "f")

        distribution = distribution_lib.DataParallel(devices=["cpu:0"])
        distribution.auto_shard_dataset = False

        with distribution.scope():
            model.fit(inputs, targets, epochs=1, batch_size=4, verbose=0)
            temp_filepath = os.path.join(
                self.get_temp_dir(), "model.weights.h5"
            )
            model.save_weights(temp_filepath)

            new_model = models.Sequential(
                [layers.Dense(8, input_shape=(4,)), layers.Dense(1)]
            )
            new_model.compile(optimizer="adam", loss="mse")
            new_model.load_weights(temp_filepath)

            for ref_w, new_w in zip(
                model.get_weights(), new_model.get_weights()
            ):
                self.assertAllClose(ref_w, new_w)

    def test_metrics_synchronization(self):
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                "gloo", init_method="tcp://127.0.0.1:0", rank=0, world_size=1
            )
            self.addCleanup(torch.distributed.destroy_process_group)

        model = models.Sequential([layers.Dense(1, input_shape=(4,))])
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])

        mesh = distribution_lib.DeviceMesh((1,), ["data"], ["cpu"])
        distribution = distribution_lib.DataParallel(device_mesh=mesh)

        with distribution.scope():
            object.__setattr__(model, "_ddp_model", model)
            model._in_ddp_context = True

            with mock.patch.object(model, "_setup_ddp"), \
                 mock.patch("torch.distributed.all_reduce") as mock_all_reduce:
                model.fit(np.ones((4, 4), "f"), np.ones((4, 1), "f"), epochs=1, verbose=0)

                mock_all_reduce.reset_mock()

                model._sync_metrics()
                self.assertGreaterEqual(mock_all_reduce.call_count, 2)

                for call in mock_all_reduce.call_args_list:
                    args, kwargs = call
                    self.assertEqual(kwargs.get("op"), torch.distributed.ReduceOp.SUM)

    def test_gradient_accumulation_no_sync(self):
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                "gloo", init_method="tcp://127.0.0.1:0", rank=0, world_size=1
            )
            self.addCleanup(torch.distributed.destroy_process_group)

        model = models.Sequential([layers.Dense(1, input_shape=(4,))])
        optimizer = optimizers.Adam(gradient_accumulation_steps=2)
        model.compile(optimizer=optimizer, loss="mse")

        mesh = distribution_lib.DeviceMesh((1,), ["data"], ["cpu"])
        distribution = distribution_lib.DataParallel(device_mesh=mesh)

        with distribution.scope():
            model.no_sync = mock.MagicMock(return_value=contextlib.nullcontext())
            object.__setattr__(model, "_ddp_model", model)
            model._in_ddp_context = True

            with mock.patch.object(model, "_setup_ddp"):
                model.train_on_batch(np.ones((4, 4), "f"), np.ones((4, 1), "f"))
                model.no_sync.assert_called_once()

                model.no_sync.reset_mock()
                model.train_on_batch(np.ones((4, 4), "f"), np.ones((4, 1), "f"))
                model.no_sync.assert_not_called()

    def test_distributed_checkpointing_rank_logic(self):
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                "gloo", init_method="tcp://127.0.0.1:0", rank=0, world_size=1
            )
            self.addCleanup(torch.distributed.destroy_process_group)

        model = models.Sequential([layers.Dense(1, input_shape=(4,))])
        model.compile(optimizer="adam", loss="mse")
        model(torch.ones((1, 4)))

        temp_filepath = os.path.join(self.get_temp_dir(), "model.weights.h5")

        with mock.patch("keras.src.backend.torch.distribution_lib.process_id", return_value=0):
            model.save_weights(temp_filepath)
            self.assertTrue(os.path.exists(temp_filepath))

        new_model = models.Sequential([layers.Dense(1, input_shape=(4,))])
        new_model.compile(optimizer="adam", loss="mse")
        new_model(torch.ones((1, 4)))

        with mock.patch("keras.src.backend.torch.distribution_lib.process_id", return_value=1):
            new_model.load_weights(temp_filepath)

        for ref_w, new_w in zip(model.get_weights(), new_model.get_weights()):
            self.assertAllClose(ref_w, new_w)
