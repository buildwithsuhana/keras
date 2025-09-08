from unittest import mock

import jax
import jax.numpy as jnp

from keras.src import layers
from keras.src import models
from keras.src.distribution import distribution_lib
from keras.src.testing import test_case


# class AutoShardDistributionTest(test_case.TestCase):
#     def setUp(self):
#         super().setUp()

#     @mock.patch(
#         "keras.src.backend.jax.trainer.JAXTrainer.jax_state_sync",
#         lambda self: None,
#     )
#     def test_autosharding(self):
#         """Tests a simple model with AutoShardDistribution."""
#         num_devices = jax.device_count()
#         if num_devices < 2:
#             self.skipTest("This test requires at least 2 devices for sharding.")

#         mesh_shape = (1, num_devices)
#         mesh = distribution_lib.DeviceMesh(
#             shape=mesh_shape, axis_names=("batch", "model")
#         )
#         distribution = distribution_lib.AutoShardDistribution(mesh)

#         with distribution.scope():
#             model = models.Sequential(
#                 [
#                     layers.Dense(16, input_shape=(8,)),
#                     layers.Dense(1),
#                 ]
#             )
#             model.build(input_shape=(None, 8))

#         sample_x = jnp.ones((2, 8), dtype=jnp.float32)
#         distribution.shard(model, sample_x)

#         model.compile(optimizer="sgd", loss="mse")

#         batch_size = num_devices * 2
#         x = jnp.ones((batch_size, 8), dtype=jnp.float32)
#         y = jnp.ones((batch_size, 1), dtype=jnp.float32)

#         with mock.patch(
#             "keras.src.trainers.trainer.Trainer.get_metrics_result",
#             return_value={"loss": 0.0},
#         ):
#             model.fit(x, y, epochs=1, steps_per_epoch=2, batch_size=batch_size)

#         for var in model.variables:
#             if "kernel" in var.path:
#                 sharding = distribution.get_variable_layout(var)
#                 self.assertEqual(
#                     sharding.spec,
#                     jax.sharding.PartitionSpec(None, None),
#                     f"Variable '{var.path}' was not replicated as expected "
#                     "under the current test mock setup. "
#                     f"Spec: {sharding.spec}",
#                 )

#     def test_tensor_is_shared_and_accessible(self):
#         num_devices = jax.device_count()
#         if num_devices < 2:
#             self.skipTest("This test requires at least 2 devices for sharding.")

#         mesh_shape = (1, num_devices)
#         axis_names = ("batch", "model")
#         mesh = distribution_lib.DeviceMesh(
#             shape=mesh_shape, axis_names=axis_names
#         )
#         distribution = distribution_lib.AutoShardDistribution(mesh)

#         with distribution.scope():
#             model = models.Sequential(
#                 [layers.Dense(4, use_bias=False, input_shape=(8,))]
#             )
#             model.build(input_shape=(None, 8))

#         kernel_var = model.variables[0]
#         self.assertIn("kernel", kernel_var.path)

#         kernel_layout = distribution.get_variable_layout(kernel_var)
#         self.assertEqual(
#             kernel_layout.spec,
#             jax.sharding.PartitionSpec(None, None),
#             "Kernel variable should be replicated across all devices.",
#         )

#         def access_on_device(dummy_arg):
#             tensor_sum = jnp.sum(kernel_var.value)
#             device_id = jax.lax.axis_index("model")

#             return tensor_sum + device_id.astype(kernel_var.dtype)

#         with distribution.scope():
#             pmapped_access_fn = jax.pmap(
#                 access_on_device,
#                 axis_name="model",
#             )

#             dummy_input = jnp.zeros(num_devices)
#             results = pmapped_access_fn(dummy_input)

#         expected_sum = jnp.sum(kernel_var.value)
#         for i in range(num_devices):
#             self.assertAllClose(
#                 results[i],
#                 expected_sum + i,
#                 msg=f"Device {i} did not compute the correct value.",
#             )

#         self.assertEqual(len(results.devices()), num_devices)

from unittest import mock

import jax
import jax.numpy as jnp
import tensorflow as tf

from keras.src import layers
from keras.src import models
from keras.src import optimizers
from keras.src import utils
from keras.src.distribution import distribution_lib
from keras.src.testing import test_case


def build_opt_model(
    vocab_size,
    num_blocks=2,
    embed_dim=128,
    num_heads=4,
    ff_dim=512,
    maxlen=100,
):
    """Builds a simplified transformer model for testing."""
    inputs = layers.Input(shape=(maxlen,), dtype="int32")
    # Embedding layer
    embedding_layer = layers.Embedding(
        input_dim=vocab_size, output_dim=embed_dim
    )
    x = embedding_layer(inputs)

    # Transformer blocks
    for _ in range(num_blocks):
        # Multi-head self-attention
        attn_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )(x, x)
        x = layers.Add()([x, attn_output])
        x = layers.LayerNormalization()(x)

        # Feed-forward network
        ffn_output = layers.Dense(ff_dim, activation="relu")(x)
        ffn_output = layers.Dense(embed_dim)(ffn_output)
        x = layers.Add()([x, ffn_output])
        x = layers.LayerNormalization()(x)

    # Output layer
    outputs = layers.Dense(vocab_size, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model


class AutoShardDistributionTest(test_case.TestCase):
    def setUp(self):
        super().setUp()

    @mock.patch(
        "keras.src.backend.jax.trainer.JAXTrainer.jax_state_sync",
        lambda self: None,
    )
    def test_autosharding(self):
        """Tests a simple model with AutoShardDistribution."""
        num_devices = jax.device_count()
        if num_devices < 2:
            self.skipTest("This test requires at least 2 devices for sharding.")

        mesh_shape = (1, num_devices)
        mesh = distribution_lib.DeviceMesh(
            shape=mesh_shape, axis_names=("batch", "model")
        )
        distribution = distribution_lib.AutoShardDistribution(mesh)

        with distribution.scope():
            model = models.Sequential(
                [
                    layers.Dense(16, input_shape=(8,)),
                    layers.Dense(1),
                ]
            )
            model.build(input_shape=(None, 8))

        sample_x = jnp.ones((2, 8), dtype=jnp.float32)
        distribution.shard(model, sample_x)

        model.compile(optimizer="sgd", loss="mse")

        batch_size = num_devices * 2
        x = jnp.ones((batch_size, 8), dtype=jnp.float32)
        y = jnp.ones((batch_size, 1), dtype=jnp.float32)

        with mock.patch(
            "keras.src.trainers.trainer.Trainer.get_metrics_result",
            return_value={"loss": 0.0},
        ):
            model.fit(x, y, epochs=1, steps_per_epoch=2, batch_size=batch_size)

        for var in model.variables:
            if "kernel" in var.path:
                sharding = distribution.get_variable_layout(var)
                self.assertEqual(
                    sharding.spec,
                    jax.sharding.PartitionSpec(None, None),
                    f"Variable '{var.path}' was not replicated as expected "
                    "under the current test mock setup. "
                    f"Spec: {sharding.spec}",
                )

    def test_tensor_is_shared_and_accessible(self):
        num_devices = jax.device_count()
        if num_devices < 2:
            self.skipTest("This test requires at least 2 devices for sharding.")

        mesh_shape = (1, num_devices)
        axis_names = ("batch", "model")
        mesh = distribution_lib.DeviceMesh(
            shape=mesh_shape, axis_names=axis_names
        )
        distribution = distribution_lib.AutoShardDistribution(mesh)

        with distribution.scope():
            model = models.Sequential(
                [layers.Dense(4, use_bias=False, input_shape=(8,))]
            )
            model.build(input_shape=(None, 8))

        kernel_var = model.variables[0]
        self.assertIn("kernel", kernel_var.path)

        kernel_layout = distribution.get_variable_layout(kernel_var)
        self.assertEqual(
            kernel_layout.spec,
            jax.sharding.PartitionSpec(None, None),
            "Kernel variable should be replicated across all devices.",
        )

        def access_on_device(dummy_arg):
            tensor_sum = jnp.sum(kernel_var.value)
            device_id = jax.lax.axis_index("model")

            return tensor_sum + device_id.astype(kernel_var.dtype)

        with distribution.scope():
            pmapped_access_fn = jax.pmap(
                access_on_device,
                axis_name="model",
            )

            dummy_input = jnp.zeros(num_devices)
            results = pmapped_access_fn(dummy_input)

        expected_sum = jnp.sum(kernel_var.value)
        for i in range(num_devices):
            self.assertAllClose(
                results[i],
                expected_sum + i,
                msg=f"Device {i} did not compute the correct value.",
            )

        self.assertEqual(len(results.devices()), num_devices)

    def test_opt125m_training_on_tiny_shakespeare(self):
        """Tests training a transformer on a text dataset."""
        num_devices = jax.device_count()
        if num_devices < 2:
            self.skipTest("This test requires at least 2 devices for sharding.")

        # 1. Setup distribution
        mesh_shape = (num_devices,)
        axis_names = ("batch",)
        mesh = distribution_lib.DeviceMesh(mesh_shape, axis_names)
        distribution = distribution_lib.AutoShardDistribution(mesh)

        # 2. Prepare dataset
        path = utils.get_file(
            "tinyshakespeare.txt",
            "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        )
        with open(path) as f:
            text = f.read()

        chars = sorted(list(set(text)))
        vocab_size = len(chars)
        char_to_int = {c: i for i, c in enumerate(chars)}
        text_as_int = [char_to_int[c] for c in text]

        maxlen = 100
        dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
        sequences = dataset.batch(maxlen + 1, drop_remainder=True)

        def split_input_target(chunk):
            return chunk[:-1], chunk[1:]

        dataset = sequences.map(split_input_target)
        batch_size = 16 * num_devices
        train_dataset = (
            dataset.shuffle(10000).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
        )

        # 3. Build and compile model within distribution scope
        with distribution.scope():
            model = build_opt_model(vocab_size=vocab_size, maxlen=maxlen)
            model.compile(
                optimizer=optimizers.Adam(learning_rate=0.001),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )

        # 4. Train the model
        history = model.fit(
            train_dataset,
            epochs=1,
            steps_per_epoch=5,  # Keep test duration short
        )

        # 5. Assert results
        self.assertIsInstance(history.history, dict)
        self.assertIn("loss", history.history)
        self.assertIn("accuracy", history.history)
        self.assertGreater(len(history.history["loss"]), 0)
        self.assertFalse(jnp.isnan(history.history["loss"][0]))
