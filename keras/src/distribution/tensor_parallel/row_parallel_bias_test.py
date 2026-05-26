import os
os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

import keras
from keras import layers
from keras import ops
import numpy as np
from keras.src.distribution.tensor_parallel.tensor_parallel import TensorParallelKeras
from keras.src import testing

class RowParallelBiasTest(testing.TestCase):
    def test_row_parallel_bias_correctness(self):
        # Create a model with a Dense layer that will be sharded row-parallel
        # A large input and small output typically triggers row-parallel (down_projection)
        inputs = keras.Input(shape=(128,))
        # 128 -> 32 is a contraction, should be row-parallel
        layer = layers.Dense(32, use_bias=True, kernel_initializer='ones', bias_initializer='ones')
        outputs = layer(inputs)
        model = keras.Model(inputs, outputs)
        
        # Verify it's row-parallel
        from keras.src.distribution.tensor_parallel.autoconfig import get_default_config
        config = get_default_config(model, ["cpu:0", "cpu:1"])
        
        layer_path = model.layers[1].path
        from keras.src.distribution.tensor_parallel.autoconfig import _reduce_sum
        self.assertEqual(config.output_rules[layer_path], _reduce_sum)
        
        # Original output
        x = np.ones((1, 128)).astype("float32")
        original_output = model(x)
        # Expected: sum(ones(128) * ones(128)) + 1 = 128 + 1 = 129
        self.assertAllClose(original_output, np.full((1, 32), 129.0))
        
        # Sharded output
        tp_model = TensorParallelKeras(model, device_count=2, device_ids=["cpu:0", "cpu:1"])
        tp_output = tp_model(x)
        
        # Correct Row-Parallel TP behavior should match original output
        self.assertAllClose(tp_output, original_output)

if __name__ == "__main__":
    testing.main()
