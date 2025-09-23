import numpy as np
import keras
from keras import layers, Model
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy
from src.tensor_parallel_keras.tensor_parallel_keras import TensorParallelKeras 

def create_simple_model(input_dim=128, output_dim=10):
    """Create a simple model for testing."""
    inputs = keras.Input(shape=(input_dim,), name="input_layer")
    x = layers.Dense(256, activation='relu', name="dense_1")(inputs)
    x = layers.Dense(256, activation='relu', name="dense_2")(x)
    outputs = layers.Dense(output_dim, activation='softmax', name="dense_3")(x)
    model = Model(inputs=inputs, outputs=outputs, name="SimpleModel")
    return model

def get_original_weights(tp_model):
    """Helper to collect and de-shard weights from a TP model for comparison."""
    return tp_model.tp_manager.original_model.get_weights()

def test_backward_pass_correctness():
    """Test that backward pass produces identical weight updates."""
    print("🧪 Testing Backward Pass Correctness")
    print("=" * 60)
    
    # 1. Setup
    devices = ["cpu:0", "cpu:1"]  # Use available devices
    input_dim = 128
    output_dim = 10
    batch_size = 32
    
    print(f"🔧 Setup:")
    print(f"  - Devices: {devices}")
    print(f"  - Input dim: {input_dim}")
    print(f"  - Output dim: {output_dim}")
    print(f"  - Batch size: {batch_size}")
    
    # Create dummy data
    np.random.seed(42)  # For reproducible results
    dummy_x = np.random.rand(batch_size, input_dim).astype("float32")
    dummy_y = np.random.randint(0, output_dim, size=(batch_size,)).astype("int32")
    dummy_y = keras.utils.to_categorical(dummy_y, output_dim)
    
    print("\n🔧 Setting up single-device model...")
    model_single = create_simple_model(input_dim, output_dim)
    initial_weights = model_single.get_weights()
    optimizer_single = Adam(learning_rate=0.001)
    loss_fn = CategoricalCrossentropy()
    model_single.compile(optimizer=optimizer_single, loss=loss_fn)
    
    # 3. Initialize Tensor Parallel model
    print("\n🔧 Setting up Tensor Parallel model...")
    model_tp_base = create_simple_model(input_dim, output_dim)
    model_tp_base.set_weights(initial_weights)  # Ensure same starting point

    tp_manager = TensorParallelKeras(model_tp_base, device_ids=devices)
    model_tp_assembled = tp_manager.build_assembled_model()
    
    optimizer_tp = Adam(learning_rate=0.001)
    model_tp_assembled.compile(optimizer=optimizer_tp, loss=loss_fn)

    print("\n🚀 Performing one training step...")
    loss_single = model_single.train_on_batch(dummy_x, dummy_y)
    loss_tp = model_tp_assembled.train_on_batch(dummy_x, dummy_y)
    
    print("\n🔍 Comparing results...")
    print(f"   - Single-device loss: {loss_single:.6f}")
    print(f"   - Tensor Parallel loss: {loss_tp:.6f}")
    
    loss_diff = abs(loss_single - loss_tp)
    assert loss_diff < 1e-5, f"Loss difference is too large: {loss_diff:.2e}"
    print(f"✅ Losses match perfectly! (difference: {loss_diff:.2e})")
    
    weights_single_updated = model_single.get_weights()
    weights_tp_updated = tp_manager.original_model.get_weights()
    
    all_weights_match = True
    for i, (w_single, w_tp) in enumerate(zip(weights_single_updated, weights_tp_updated)):
        try:
            np.testing.assert_allclose(w_single, w_tp, rtol=1e-5, atol=1e-5)
            print(f"   ✅ Weight {i} ({w_single.shape}) matches.")
        except AssertionError:
            print(f"   ❌ Weight {i} ({w_single.shape}) MISMATCH!")
            all_weights_match = False
            
    if all_weights_match:
        print("\n🎉 BACKWARD PASS TEST PASSED!")
        return True
    else:
        print("\n❌ BACKWARD PASS TEST FAILED!")
        return False

if __name__ == "__main__":
    if test_backward_pass_correctness():
        print("\n🏆 ALL TESTS PASSED! Backward pass is correct.")
        exit(0)
    else:
        print("\n❌ A TEST FAILED!")
        exit(1)