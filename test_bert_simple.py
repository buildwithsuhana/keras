"""Simple test for non-floating dtype fix using BERT backbone with TorchLayer"""

import os
# Must be set before any other imports
os.environ["KERAS_BACKEND"] = "torch"

import torch
import keras
import keras_hub

# Quick test without distributed training
def test_bert_layer_non_float_dtypes():
    """Test that BERT layers with non-float dtypes work correctly."""
    
    print("Loading BERT-tiny backbone...")
    backbone = keras_hub.models.BertBackbone.from_preset("bert_tiny_en_uncased")
    
    print(f"Backbone loaded: {backbone.summary()}")
    
    # Test forward pass with dummy data
    import numpy as np
    
    batch_size = 2
    seq_length = 32
    
    # Create dummy token IDs and attention mask
    token_ids = np.random.randint(0, 1000, size=(batch_size, seq_length))
    attention_mask = np.ones((batch_size, seq_length))
    
    print(f"Input shapes: token_ids={token_ids.shape}, mask={attention_mask.shape}")
    
    # Run forward pass - this tests TorchLayer with various dtype parameters
    print("Running forward pass...")
    output = backbone({"token_ids": token_ids, "attention_mask": attention_mask})
    
    print(f"Output shape: {output['sequence_output'].shape}")
    
    # Test that we can access all variables without RuntimeError
    print("\nChecking variables for non-float dtype issues...")
    for i, variable in enumerate(backbone.variables):
        var_tensor = variable.value
        if hasattr(var_tensor, 'dtype'):
            is_float = var_tensor.dtype.is_floating_point or var_tensor.dtype.is_complex
            param_type = "Parameter" if isinstance(var_tensor, torch.nn.Parameter) else "Tensor"
            print(f"  Variable {i}: dtype={var_tensor.dtype}, is_float={is_float}, type={param_type}")
    
    # Verify named_parameters only contains float tensors
    print("\nChecking named_parameters...")
    param_count = 0
    for name, param in backbone.named_parameters():
        if hasattr(param, 'dtype'):
            assert param.dtype.is_floating_point, f"Parameter '{name}' has non-float dtype: {param.dtype}"
        param_count += 1
    
    print(f"  Total parameters: {param_count}")
    print("  ✓ All named_parameters have float dtypes (as expected)")
    
    print("\n" + "="*50)
    print("✓ BERT-tiny test passed!")
    print("✓ Non-floating dtype fix is working correctly!")
    print("="*50)


if __name__ == "__main__":
    test_bert_layer_non_float_dtypes()

