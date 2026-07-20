import os

# Force virtual devices for JAX
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
os.environ["KERAS_BACKEND"] = "jax"

import numpy as np
import pytest

from keras.src import backend
from keras.src import testing
from keras.src.backend import distribution_lib as backend_dlib
from keras.src.distribution import distribution_lib
from keras.src.distribution.distribution_lib import ParallaxDistribution


@pytest.mark.skipif(
    backend.backend() != "jax",
    reason="ParallaxDistribution currently only supports JAX backend",
)
class ParallaxDistributionTest(testing.TestCase):
    def test_init_with_custom_mesh(self):
        devices = backend_dlib.list_devices()
        self.assertEqual(len(devices), 8)

        # 1D Mesh
        mesh = distribution_lib.DeviceMesh(
            shape=(8,), axis_names=["model"], devices=devices
        )

        dist = ParallaxDistribution(strategy="fsdp", mesh=mesh)
        self.assertIs(dist.device_mesh, mesh)
        self.assertEqual(dist.device_mesh.shape, (8,))
        self.assertEqual(dist.device_mesh.axis_names, ["model"])
        self.assertEqual(dist.batch_dim_name, "model")  # Fallback to first axis

    def test_get_variable_layout_fsdp_divisible(self):
        devices = backend_dlib.list_devices()
        mesh = distribution_lib.DeviceMesh(
            shape=(8,), axis_names=["model"], devices=devices
        )
        dist = ParallaxDistribution(strategy="fsdp", mesh=mesh)

        # Divisible dimension (8)
        variable = backend.Variable(initializer=np.ones((16, 8)))
        layout = dist.get_variable_layout(variable)
        self.assertEqual(layout.axes, (None, "model"))

    def test_get_variable_layout_fsdp_indivisible(self):
        devices = backend_dlib.list_devices()
        mesh = distribution_lib.DeviceMesh(
            shape=(8,), axis_names=["model"], devices=devices
        )
        dist = ParallaxDistribution(strategy="fsdp", mesh=mesh)

        # Indivisible dimension (2)
        variable = backend.Variable(initializer=np.ones((16, 2)))
        layout = dist.get_variable_layout(variable)
        # Should fallback to replication (None, None)
        self.assertEqual(layout.axes, (None, None))

    def test_get_variable_layout_fsdp_no_model_axis(self):
        devices = backend_dlib.list_devices()
        mesh = distribution_lib.DeviceMesh(
            shape=(8,), axis_names=["data"], devices=devices
        )
        dist = ParallaxDistribution(strategy="fsdp", mesh=mesh)

        variable = backend.Variable(initializer=np.ones((16, 8)))
        layout = dist.get_variable_layout(variable)
        # Should fallback to replication because "model" axis is missing
        self.assertEqual(layout.axes, (None, None))
