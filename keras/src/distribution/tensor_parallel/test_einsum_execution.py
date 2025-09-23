#!/usr/bin/env python3
"""
Test EinsumDense tensor parallelism execution
"""

import os
import numpy as np

# 💻 Simulate 2 CPU devices for JAX BEFORE importing jax
# os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=2'

# import jax
# print(f"🔍 JAX Device Detection:")
# print(f"   Number of JAX devices: {jax.local_device_count()}")
# print(f"   Device list: {jax.devices()}")
# print(f"   Device types: {[str(d) for d in jax.devices()]}")

import keras
from keras.layers import Input, EinsumDense, Dense
from src.tensor_parallel_keras.tensor_parallel_keras import TensorParallelKeras

def create_einsum_model():
    """Create a simple EinsumDense model."""
    inputs = Input(shape=(10, 64), name='input_tensor')
    
    # EinsumDense layer with equation "btd,de->bte" (input projection)
    einsum_output = EinsumDense(
        equation='btd,de->bte',
        output_shape=(None, 32),
        name='einsum_dense'
    )(inputs)
    
    # Final output projection
    output = Dense(16, activation='linear', name='output_dense')(einsum_output)
    
    model = keras.Model(inputs=inputs, outputs=output)
    return model

def test_einsum_execution():
    """Test EinsumDense tensor parallelism execution."""
    print("\n🧪 Testing EinsumDense tensor parallelism execution...")
    
    # Create model
    model = create_einsum_model()
    print(f"   - Model created with {len(model.layers)} layers")
    
    # Create tensor parallel manager
    tp_manager = TensorParallelKeras(
        model=model,
        device_ids=['cpu:0', 'cpu:1']
    )
    print("   - Tensor parallel manager created")

    # Build the final, ASSEMBLED model from the manager
    model_tp_assembled = tp_manager.build_assembled_model()
    print("   - Assembled tensor parallel model built")
    
    # Create test input
    input_data = np.random.random((8, 10, 64)).astype(np.float32)
    print(f"   - Input data shape: {input_data.shape}")
    
    # Run single device model
    single_output = model(input_data)
    print(f"\n▶️ Running models...")
    print(f"   - Single device output shape: {single_output.shape}")
    
    # Run tensor parallel model
    tp_output = model_tp_assembled(input_data)
    print(f"   - Tensor parallel output shape: {tp_output.shape}")
    
    # Check shapes match
    shape_match = single_output.shape == tp_output.shape
    print(f"\n🔍 Comparing outputs...")
    print(f"   - Shape match: {shape_match}")
    
    if shape_match:
        # --- MODIFICATION START ---
        # Use the safe, backend-agnostic Keras function to convert tensors
        # from any device (CPU, GPU, MPS) to NumPy arrays.
        print("   - Converting outputs to NumPy for comparison...")
        single_np = keras.ops.convert_to_numpy(single_output)
        tp_np = keras.ops.convert_to_numpy(tp_output)
        # --- MODIFICATION END ---
        
        # Calculate differences
        abs_diff = np.abs(single_np - tp_np)
        rel_diff = abs_diff / (np.abs(single_np) + 1e-8)
        
        print(f"   - Max absolute difference: {np.max(abs_diff):.2e}")
        print(f"   - Max relative difference: {np.max(rel_diff):.2e}")
        
        # Check if within tolerance
        tolerance = 1e-5
        within_tolerance = np.max(abs_diff) < tolerance
        
        if within_tolerance:
            print("\n✅ MATHEMATICAL IDENTITY ACHIEVED! (within tolerance)")
        else:
            print("\n❌ Mathematical differences detected")
            
        # Show sample values
        print("\n   Sample values (first 5 elements of first batch item):")
        print(f"     Single device:   {single_np[0, 0, :5]}")
        print(f"     Tensor parallel: {tp_np[0, 0, :5]}")
        print(f"     Differences:     {abs_diff[0, 0, :5]}")
    else:
        print("\n❌ Shape mismatch - execution failed")

if __name__ == "__main__":
    test_einsum_execution()
