#!/usr/bin/env python3
"""
Test simple Dense layer tensor parallelism execution
"""

import os
import numpy as np

# 💻 Set this flag BEFORE importing jax
# os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=2'

# import jax
# print(f"🔍 JAX Device Detection:")
# print(f"   Number of JAX devices: {jax.local_device_count()}")
# print(f"   Device list: {jax.devices()}")

import keras
from keras.layers import Input, Dense
from src.tensor_parallel_keras.tensor_parallel_keras import TensorParallelKeras

def create_simple_model():
    """Create a simple Dense model."""
    inputs = Input(shape=(64,), name='input_tensor')
    output = Dense(32, activation='relu', name='dense')(inputs)
    model = keras.Model(inputs=inputs, outputs=output)
    return model

def test_dense_execution():
    """Test simple Dense layer tensor parallelism execution."""
    print("\n🧪 Testing simple Dense layer tensor parallelism execution...")
    
    model = create_simple_model()
    print(f"   - Model created with {len(model.layers)} layers")
    
    tp_manager = TensorParallelKeras(
        model=model,
        device_ids=['cpu:0', 'cpu:1']
    )
    print("   - Tensor parallel manager created")

    model_tp_assembled = tp_manager.build_assembled_model()
    print("   - Assembled tensor parallel model built")
    
    input_data = np.random.random((8, 64)).astype(np.float32)
    print(f"   - Input data shape: {input_data.shape}")
    
    single_output = model(input_data)
    print(f"\n▶️ Running models...")
    print(f"   - Single device output shape: {single_output.shape}")
    
    tp_output = model_tp_assembled(input_data)
    print(f"   - Tensor parallel output shape: {tp_output.shape}")
    
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
        
        abs_diff = np.abs(single_np - tp_np)
        print(f"   - Max absolute difference: {np.max(abs_diff):.2e}")
        
        tolerance = 1e-5
        within_tolerance = np.max(abs_diff) < tolerance
        
        if within_tolerance:
            print("\n✅ MATHEMATICAL IDENTITY ACHIEVED! (within tolerance)")
        else:
            print("\n❌ Mathematical differences detected")
            
        print("\n   Sample values (first 5 elements of first batch item):")
        print(f"     Single device:   {single_np[0, :5]}")
        print(f"     Tensor parallel: {tp_np[0, :5]}")
    else:
        print("\n❌ Shape mismatch - execution failed")

if __name__ == "__main__":
    test_dense_execution()