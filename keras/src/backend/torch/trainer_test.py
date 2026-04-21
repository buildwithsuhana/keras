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
