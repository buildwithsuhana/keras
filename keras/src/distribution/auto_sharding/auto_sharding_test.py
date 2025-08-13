from unittest import mock

import jax
import numpy as np
import tensorflow as tf

from keras.src import layers
from keras.src import models
from keras.src.distribution import distribution_lib
from keras.src.distribution.auto_sharding.api import AutoShardDistribution
from keras.src.distribution.auto_sharding.core_components import ShardingPlan
from keras.src.testing import test_case


class AutoShardDistributionTest(test_case.TestCase):
    def test_deferred_planning_on_distribute_dataset(self):
        print(
            "\n Testing Deferred Planning Workflow (during distribute_dataset)"
        )

        device_mesh = distribution_lib.DeviceMesh(
            shape=(4,), axis_names=["batch"]
        )

        distribution = AutoShardDistribution(device_mesh)

        mock_model = mock.MagicMock(spec=models.Model)
        mock_model.input_spec = [
            layers.InputSpec(shape=(None, 4), dtype="float32")
        ]

        def simple_call(*args, **kwargs):
            return jax.ShapeDtypeStruct(shape=(2, 8), dtype=np.float32)

        mock_model.call = simple_call

        mock_dataset = mock.MagicMock()
        mock_dataset.element_spec = (
            tf.TensorSpec(shape=(2, 4), dtype=tf.float32),
            tf.TensorSpec(shape=(2, 8), dtype=tf.float32),
        )

        distribution.build(mock_model)

        with (
            mock.patch(
                "keras.src.distribution.distribution_lib.Distribution.distribute_dataset"
            ),
            mock.patch("jax.make_jaxpr") as mock_make_jaxpr,
            mock.patch(
                "keras.src.backend.jax.auto_sharding.planner.JaxShardingPlanner.plan"
            ) as mock_plan,
            mock.patch(
                "keras.src.backend.jax.auto_sharding.applier.JaxShardApplier.apply"
            ) as mock_apply,
        ):
            valid_layout_map = distribution_lib.LayoutMap(
                device_mesh=device_mesh
            )
            mock_plan.return_value = ShardingPlan(layout_map=valid_layout_map)

            distribution.distribute_dataset(mock_dataset)

            mock_make_jaxpr.assert_called_once()
            mock_plan.assert_called_once()
            mock_apply.assert_called_once()

        print("\n✅ Test Passed: Deferred auto-sharding workflow is verified!")

    def test_eager_planning_on_build(self):
        print("\n🧪 Testing Eager Planning Workflow (during build)")
        device_mesh = distribution_lib.DeviceMesh(
            shape=(4,), axis_names=["batch"]
        )
        distribution = AutoShardDistribution(device_mesh)

        mock_model = mock.MagicMock(spec=models.Model)

        mock_model.input_spec = [tf.TensorSpec(shape=(2, 4), dtype=tf.float32)]

        def simple_call(*args, **kwargs):
            return jax.ShapeDtypeStruct(shape=(2, 8), dtype=np.float32)

        mock_model.call = simple_call

        with (
            mock.patch("jax.make_jaxpr") as mock_make_jaxpr,
            mock.patch(
                "keras.src.backend.jax.auto_sharding.planner.JaxShardingPlanner.plan"
            ) as mock_plan,
            mock.patch(
                "keras.src.backend.jax.auto_sharding.applier.JaxShardApplier.apply"
            ) as mock_apply,
        ):
            valid_layout_map = distribution_lib.LayoutMap(
                device_mesh=device_mesh
            )
            mock_plan.return_value = ShardingPlan(layout_map=valid_layout_map)

            distribution.build(mock_model)

            mock_make_jaxpr.assert_called_once()
            mock_plan.assert_called_once()
            mock_apply.assert_called_once()

        print("\nTest Passed: Eager auto-sharding workflow is verified!")
