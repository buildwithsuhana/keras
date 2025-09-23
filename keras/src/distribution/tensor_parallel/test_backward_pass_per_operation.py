#!/usr/bin/env python3
"""
Test backward pass mathematical identity for each operation separately.
This script verifies that weight updates are identical between a single-device
model and its tensor-parallel equivalent for various layer types.
"""

import os
# Ensure JAX backend is set before importing Keras
os.environ["KERAS_BACKEND"] = "jax"

# Simulate 2 CPU devices for JAX. This MUST be set before JAX initializes.
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=2'

import jax
import numpy as np
import keras
from keras import layers, Model
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy
from src.tensor_parallel_keras.tensor_parallel_keras import TensorParallelKeras

# --- Model Creation Functions (Unchanged) ---

def create_dense_model(input_dim=64, output_dim=32):
    """Create a simple Dense layer model."""
    inputs = keras.Input(shape=(input_dim,))
    x = layers.Dense(output_dim, activation='relu', name="dense")(inputs)
    outputs = layers.Dense(16, activation='softmax', name="output")(x)
    model = Model(inputs=inputs, outputs=outputs, name="DenseModel")
    return model

def create_mlp_model(input_dim=32, hidden_dim=128, output_dim=16):
    """Create an MLP model with up/down projections."""
    inputs = keras.Input(shape=(input_dim,))
    x = layers.Dense(hidden_dim, activation='relu', name="dense_up")(inputs)
    x = layers.Dense(hidden_dim // 2, activation='relu', name="dense_down")(x)
    outputs = layers.Dense(output_dim, activation='softmax', name="output_dense")(x)
    model = Model(inputs=inputs, outputs=outputs, name="MLPModel")
    return model

def create_einsum_model(input_dim=128, hidden_dim=512, output_dim=128):
    """Create a model with EinsumDense layers."""
    # Define the sequence length to avoid shape inference issues with 'None'
    sequence_length = 10
    inputs = keras.Input(shape=(sequence_length, input_dim))
    x = layers.EinsumDense(
        equation='btd,de->bte',
        # Explicitly provide the sequence length in the output shape
        output_shape=(sequence_length, hidden_dim),
        activation='relu',
        name="einsum_dense"
    )(inputs)
    x = layers.EinsumDense(
        equation='bte,de->btd',
        # Explicitly provide the sequence length here as well
        output_shape=(sequence_length, output_dim),
        activation='relu',
        name="einsum_dense_1"
    )(x)
    # Flatten the sequence for CategoricalCrossentropy
    x = layers.Flatten()(x)
    outputs = layers.Dense(16, activation='softmax', name="output")(x)
    model = Model(inputs=inputs, outputs=outputs, name="EinsumModel")
    return model

def create_mha_model(input_dim=32, num_heads=4):
    """Create a model with Multi-Head Attention."""
    inputs = keras.Input(shape=(10, input_dim))
    x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=input_dim, name="mha")(inputs, inputs)
    # Flatten the sequence for CategoricalCrossentropy
    x = layers.Flatten()(x)
    outputs = layers.Dense(16, activation='softmax', name="output")(x)
    model = Model(inputs=inputs, outputs=outputs, name="MHAModel")
    return model

def create_embedding_model(vocab_size=1000, embedding_dim=64, output_dim=32):
    """Create a model with Embedding layer."""
    inputs = keras.Input(shape=(10,), dtype='int32')
    x = layers.Embedding(vocab_size, embedding_dim, name="embedding")(inputs)
    # Flatten the sequence for CategoricalCrossentropy
    x = layers.Flatten()(x)
    outputs = layers.Dense(16, activation='softmax', name="output")(x)
    model = Model(inputs=inputs, outputs=outputs, name="EmbeddingModel")
    return model

# --- Main Test Function (Corrected Architecture) ---

def test_backward_pass_identity(model_creator, model_name, input_shape, is_sequential=False):
    """Test backward pass mathematical identity for a specific model type."""
    print(f"\n🧪 Testing {model_name} Backward Pass")
    print("=" * 60)
    
    # Initialize test status flags
    all_tests_passed = True

    # Create data
    np.random.seed(42)
    if "Embedding" in model_name:
        dummy_x = np.random.randint(0, 1000, size=input_shape).astype("int32")
    else:
        dummy_x = np.random.rand(*input_shape).astype("float32")

    # Target shape is always (batch_size,) for labels before one-hot encoding
    target_shape = (input_shape[0],)
    dummy_y_labels = np.random.randint(0, 16, size=target_shape).astype("int32")
    dummy_y = keras.utils.to_categorical(dummy_y_labels, 16)
    
    print(f"🔧 Setup: JAX backend with {len(jax.devices())} devices. Input shape: {dummy_x.shape}")
    
    # --- Single-Device Model ---
    print(f"\n🔧 Setting up single-device {model_name}...")
    model_single = model_creator()
    optimizer_single = Adam(learning_rate=0.001)
    loss_fn = CategoricalCrossentropy()
    model_single.compile(optimizer=optimizer_single, loss=loss_fn)
    initial_weights = model_single.get_weights()
    
    # --- Tensor Parallel Model ---
    print(f"\n🔧 Setting up Tensor Parallel {model_name}...")
    model_tp_base = model_creator()
    model_tp_base.set_weights(initial_weights)

    # MODIFICATION: Use the manager/builder pattern
    # 1. Create the manager
    tp_manager = TensorParallelKeras(model=model_tp_base, device_ids=['cpu:0', 'cpu:1'])
    # 2. Build the final, trainable model from the manager
    model_tp_assembled = tp_manager.build_assembled_model()

    optimizer_tp = Adam(learning_rate=0.001)
    model_tp_assembled.compile(optimizer=optimizer_tp, loss=loss_fn)
    
    # --- Run Tests ---
    
    # Test 1: Forward Pass Identity
    print(f"\n[1/3] 🔍 Testing Forward Pass Mathematical Identity...")
    try:
        pred_single = model_single.predict(dummy_x, verbose=0)
        pred_tp = model_tp_assembled.predict(dummy_x, verbose=0)
        np.testing.assert_allclose(pred_single, pred_tp, rtol=1e-5, atol=1e-5)
        print(f"   ✅ PASSED: Forward pass outputs are identical.")
    except Exception as e:
        print(f"   ❌ FAILED: Forward pass failed: {e}")
        all_tests_passed = False

    # Test 2: Training & Loss Identity
    print(f"\n[2/3] 🔍 Testing Training Pipeline & Loss Identity...")
    try:
        loss_single = model_single.train_on_batch(dummy_x, dummy_y)
        loss_tp = model_tp_assembled.train_on_batch(dummy_x, dummy_y)
        
        print(f"   - Single-device loss: {loss_single:.6f}")
        print(f"   - Tensor Parallel loss: {loss_tp:.6f}")
        
        np.testing.assert_allclose(loss_single, loss_tp, rtol=1e-5, atol=1e-5)
        print(f"   ✅ PASSED: Losses are mathematically identical.")
    except Exception as e:
        print(f"   ❌ FAILED: Training pipeline or loss check failed: {e}")
        all_tests_passed = False

    # Test 3: Weight Update Identity
    print(f"\n[3/3] 🔍 Testing Weight Update Identity...")
    try:
        weights_single_updated = model_single.get_weights()
        # The manager's original_model is updated by reference, so we can get the full weights from there
        weights_tp_updated = tp_manager.original_model.get_weights()

        for i, (w_single, w_tp) in enumerate(zip(weights_single_updated, weights_tp_updated)):
            np.testing.assert_allclose(
                w_single, w_tp, rtol=1e-5, atol=1e-5,
                err_msg=f"Weight at index {i} did not match after training."
            )
        print(f"   ✅ PASSED: All weight tensors match perfectly after one training step.")
    except Exception as e:
        print(f"   ❌ FAILED: Weight comparison failed: {e}")
        all_tests_passed = False

    # Overall test result
    if all_tests_passed:
        print(f"\n🎉 {model_name} BACKWARD PASS TEST PASSED!")
    else:
        print(f"\n❌ {model_name} BACKWARD PASS TEST FAILED!")
    return all_tests_passed

# --- Test Runner ---

def run_all_backward_pass_tests():
    """Run backward pass tests for all operation types."""
    print("🧪 COMPREHENSIVE BACKWARD PASS TESTING BY OPERATION")
    print("=" * 80)
    
    test_results = {}
    
    test_results['Dense'] = test_backward_pass_identity(
        create_dense_model, "Dense Layer", (32, 64)
    )
    test_results['MLP'] = test_backward_pass_identity(
        create_mlp_model, "MLP Model", (32, 32)
    )
    # Note: Sequential models are flattened before the final Dense layer to match target shape
    test_results['EinsumDense'] = test_backward_pass_identity(
        create_einsum_model, "EinsumDense Model", (32, 10, 128), is_sequential=True
    )
    test_results['MultiHeadAttention'] = test_backward_pass_identity(
        create_mha_model, "Multi-Head Attention Model", (32, 10, 32), is_sequential=True
    )
    test_results['Embedding'] = test_backward_pass_identity(
        create_embedding_model, "Embedding Model", (32, 10), is_sequential=True
    )
    
    # --- Summary ---
    print("\n" + "=" * 80)
    print("🏆 BACKWARD PASS TESTING COMPLETED!")
    print("=" * 80)
    
    passed = sum(test_results.values())
    total = len(test_results)
    
    print("📊 RESULTS SUMMARY:")
    for operation, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  - {operation}: {status}")
    
    print(f"\n📈 OVERALL: {passed}/{total} PASSED ({(passed/total)*100:.1f}%)")
    
    if passed == total:
        print(f"\n🎉 ALL OPERATIONS PASSED BACKWARD PASS TESTING!")
        exit(0)
    else:
        print(f"\n⚠️ SOME OPERATIONS FAILED BACKWARD PASS TESTING!")
        exit(1)

if __name__ == "__main__":
    run_all_backward_pass_tests()
