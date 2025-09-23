#!/usr/bin/env python3
"""
Test Embedding tensor parallelism execution
"""

import numpy as np
import keras
from keras.layers import Input, Embedding, GlobalAveragePooling1D, Dense
from src.tensor_parallel_keras.tensor_parallel_keras import TensorParallelKeras

def create_embedding_model():
    """Create a simple Embedding model."""
    inputs = Input(shape=(10,), name='input_tensor')
    
    # Embedding layer
    embedding_output = Embedding(
        input_dim=1000, 
        output_dim=64, 
        name='embedding'
    )(inputs)
    
    # Global average pooling
    pooled = GlobalAveragePooling1D(name='global_average_pooling1d')(embedding_output)
    
    # Final output
    output = Dense(32, activation='linear', name='output_dense')(pooled)
    
    model = keras.Model(inputs=inputs, outputs=output)
    return model

def test_embedding_execution():
    """Test Embedding tensor parallelism execution."""
    print("\n🧪 Testing Embedding tensor parallelism execution...")
    
    # Create model
    model = create_embedding_model()
    print(f"   - Model created with {len(model.layers)} layers")
    
    # Create tensor parallel model
    tp_manager = TensorParallelKeras(
        model=model,
        device_ids=['cpu:0', 'cpu:1']
    )
    print("   - Tensor parallel manager created")

    # Build the final, ASSEMBLED model from the manager
    model_tp_assembled = tp_manager.build_assembled_model()
    print("   - Assembled tensor parallel model built")
    
    # Create test input (integer indices for embedding)
    input_data = np.random.randint(0, 1000, size=(8, 10)).astype(np.int32)
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
        print(f"     Single device:   {single_np[0, :5]}")
        print(f"     Tensor parallel: {tp_np[0, :5]}")
        print(f"     Differences:     {abs_diff[0, :5]}")
    else:
        print("\n❌ Shape mismatch - execution failed")

if __name__ == "__main__":
    test_embedding_execution()
