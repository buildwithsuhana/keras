# Keep your os.environ setup in the terminal command, not in the file.
import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
import keras
from keras import layers
import numpy as np

# Import the class to be tested
from tensor_parallel_keras import TensorParallelKeras
# Import the Keras testing base class
from keras.src import testing


class TensorParallelKerasTest(testing.TestCase):
    """
    Test suite for the TensorParallelKeras class running on the JAX backend.
    """

    def setUp(self):
        """Set up a reusable model and data for all tests."""
        super().setUp()
        
        # --- MISSING CODE RESTORED: Model and Data Definition ---
        inputs = keras.Input(shape=(64,), name="input_layer")
        x = layers.Dense(128, activation="relu", name="dense_column_sharded")(inputs)
        outputs = layers.Dense(10, name="dense_row_sharded")(x)
        self.original_model = keras.Model(
            inputs=inputs, outputs=outputs, name="test_mlp"
        )
        
        self.input_data = np.random.rand(32, 64).astype("float32")
        self.target_data = np.random.rand(32, 10).astype("float32")
        # --- END MISSING CODE RESTORED ---
        
        
        # --- Dynamic Device Setup (from your previous refactoring) ---
        all_available_devices = keras.distribution.list_devices()
        
        # Set device_count to the minimum required (2) or what's available
        self.device_count = min(2, len(all_available_devices))
        
        # Use the actual device IDs
        self.device_ids = all_available_devices[:self.device_count]
        
        # Check if we can run distributed tests
        self.can_run_distributed = self.device_count > 1
        
        # The print statement now reflects the dynamic check
        print(f"\n✅ Configured test environment for {keras.backend.backend()} with device_count={self.device_count}.")
        print(f"✅ Distributed tests: {self.can_run_distributed}")
        # --- End Dynamic Device Setup ---
        
    def test_initialization_and_sharding_verification(self):
        """
        Tests if the model is correctly initialized and that parameter sharding occurs.
        """
        tp_model = TensorParallelKeras(
            self.original_model, 
            device_count=self.device_count, 
            device_ids=self.device_ids
        )

        # Apply conditional logic based on actual device count
        if self.can_run_distributed:
            self.assertTrue(tp_model.distributed)
            # The next two assertions check actual sharding which only happens if distributed is True
            original_params = self.original_model.count_params()
            shard_0_params = tp_model.model_shards[0].count_params()
            self.assertLess(shard_0_params, original_params, "Sharding should reduce parameters on a single shard.")
        else:
            self.assertFalse(tp_model.distributed)
        
        self.assertEqual(tp_model.device_count, self.device_count)
        self.assertEqual(len(tp_model.model_shards), self.device_count)
        
        # This check is valid regardless of distributed status as it checks the total sum of weights
        tp_model_total_params = sum(np.prod(v.shape) for v in tp_model.weights)
        self.assertEqual(tp_model_total_params, self.original_model.count_params())


    def test_non_distributed_case_device_count_one(self):
        """
        Tests the behavior when device_count is 1, ensuring it gracefully degrades
        to a standard, non-distributed model.
        """
        # Note: This test explicitly requests device_count=1, so it should always pass the non-distributed checks.
        tp_model = TensorParallelKeras(self.original_model, device_count=1)

        self.assertFalse(tp_model.distributed)
        self.assertEqual(tp_model.device_count, 1)
        self.assertEqual(len(tp_model.model_shards), 1)
        self.assertIs(tp_model.assembled_model, self.original_model)

        output = tp_model.predict(self.input_data, verbose=0)
        self.assertEqual(output.shape, (32, 10))

    def test_forward_pass_correctness(self):
        """
        Tests if the output of the sharded model is numerically identical
        to the original model.
        """
        # Re-define model locally to ensure clean weight initialization
        inputs = keras.Input(shape=(64,), name="input_layer")
        x = layers.Dense(128, activation="relu", kernel_initializer='glorot_uniform')(inputs)
        outputs = layers.Dense(10, kernel_initializer='glorot_uniform')(x)
        original_model = keras.Model(inputs=inputs, outputs=outputs)
        
        input_data = np.random.rand(32, 64).astype("float32")

        original_output = original_model(input_data, training=False)

        tp_model = TensorParallelKeras(
            original_model, 
            device_count=self.device_count, 
            device_ids=self.device_ids
        )
        
        tp_output = tp_model(input_data, training=False)

        self.assertAllClose(original_output, tp_output, atol=1e-5, rtol=1e-5)

    def test_distributed_training_workflow(self):
        """
        Tests if the model can be compiled and trained for one step without errors.
        """
        tp_model = TensorParallelKeras(
            self.original_model, 
            device_count=self.device_count, 
            device_ids=self.device_ids
        )

        # Apply conditional logic before assertion
        if not self.can_run_distributed:
            # If running on a single device, skip the distributed-specific checks
            self.skipTest(f"Skipping distributed test because device_count={self.device_count} (must be > 1)")
            
        tp_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.01),
            loss="mse",
        )
        
        # This assertion should now only be hit when we are confident the model is distributed
        self.assertTrue(hasattr(tp_model, "coordinated_optimizer"), "Coordinated optimizer should be created in distributed mode.")

        history = tp_model.fit(
            self.input_data, self.target_data, epochs=1, batch_size=16, verbose=0
        )

        self.assertIn("loss", history.history)
        self.assertIsNotNone(history.history["loss"][0])
