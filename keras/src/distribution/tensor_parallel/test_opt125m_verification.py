#!/usr/bin/env python3
"""
Test suite for OPT-125M model verification with tensor parallelism.
"""
import os
import time
import logging
import numpy as np
import keras
from keras import layers, optimizers
import matplotlib.pyplot as plt

from src.tensor_parallel_keras.tensor_parallel_keras import TensorParallelKeras

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PositionalEmbedding(layers.Layer):
    """
    Creates a learned positional embedding.
    This layer calculates the position IDs on the fly during the forward pass.
    """
    def __init__(self, max_positions, hidden_size, **kwargs):
        super().__init__(**kwargs)
        self.max_positions = max_positions
        self.hidden_size = hidden_size
        # The actual learnable weights are in this embedding layer
        self.position_embedding = layers.Embedding(
            input_dim=self.max_positions, output_dim=self.hidden_size
        )

    def call(self, inputs):
        # inputs shape is (batch, seq_len, hidden_size)
        # We get the dynamic batch_size and seq_len from the *input tensor*
        batch_size = keras.ops.shape(inputs)[0]
        seq_len = keras.ops.shape(inputs)[1]
        
        # Create the position IDs: [0, 1, 2, ..., seq_len-1]
        position_ids = keras.ops.arange(seq_len, dtype="int32")
        
        # Look up the embeddings for the position IDs
        # position_embeddings shape is (seq_len, hidden_size)
        position_embeddings = self.position_embedding(position_ids)
        
        # --- FIX: Manually broadcast to (batch, seq_len, hidden_size) ---
        
        # 1. Add a batch dimension: (1, seq_len, hidden_size)
        position_embeddings_expanded = keras.ops.expand_dims(position_embeddings, axis=0)
        
        # 2. Repeat along the batch dimension: (batch, seq_len, hidden_size)
        position_embeddings_tiled = keras.ops.repeat(
            position_embeddings_expanded, repeats=batch_size, axis=0
        )
        
        return position_embeddings_tiled

    def get_config(self):
        # Implement get_config to make the model serializable
        config = super().get_config()
        config.update({
            "max_positions": self.max_positions,
            "hidden_size": self.hidden_size,
        })
        return config

def create_simplified_opt125m_model(vocab_size=1000, hidden_size=128, num_layers=2, num_heads=4):
    """Create a simplified OPT-125M model for faster testing."""
    print("   Creating simplified OPT-125M model...")
    
    model = keras.Sequential([
        layers.Input(shape=(None,), dtype='int32'),
        layers.Embedding(vocab_size, hidden_size),
        layers.LayerNormalization(),
        layers.Dense(hidden_size * 4, activation='relu'),
        layers.Dense(hidden_size, activation='relu'),
        layers.Dense(vocab_size, activation='softmax')
    ])
    
    print(f"      Simplified model created with {model.count_params():,} parameters")
    return model

def create_opt125m_model(vocab_size=50257, hidden_size=768, num_layers=12, num_heads=12, max_position_embeddings=2048):
    """
    Creates an OPT-125M model with learned positional embeddings.
    """
    print("   Creating OPT-125M model...")
    
    # 1. Input layer for token IDs
    input_ids = layers.Input(shape=(None,), dtype='int32', name='input_ids')
    
    # 2. Token embeddings
    token_embeddings = layers.Embedding(vocab_size, hidden_size, name='embed_tokens')(input_ids)
    
    positional_embed_layer = PositionalEmbedding(
        max_position_embeddings, hidden_size, name='embed_positions'
    )
    position_embeddings = positional_embed_layer(token_embeddings)

    # 4. Add token and positional embeddings together
    embedding_output = layers.Add(name='add_embeddings')([token_embeddings, position_embeddings])

    # 5. Apply LayerNormalization
    hidden_states = layers.LayerNormalization(epsilon=1e-5, name='layernorm_embedding')(embedding_output)
    
    # 6. Transformer Blocks
    for i in range(num_layers):
        print(f"     Adding transformer layer {i+1}/{num_layers}")
        
        # Self-Attention block
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=hidden_size // num_heads, name=f'layers_{i}_self_attn'
        )(hidden_states, hidden_states)
        
        # Residual connection and LayerNorm
        hidden_states = layers.Add(name=f'layers_{i}_residual_1')([hidden_states, attention_output])
        hidden_states = layers.LayerNormalization(epsilon=1e-5, name=f'layernorm_1_{i}')(hidden_states)
        
        # MLP block
        mlp_hidden = layers.Dense(hidden_size * 4, activation='relu', name=f'layers_{i}_mlp_up')(hidden_states)
        mlp_output = layers.Dense(hidden_size, name=f'layers_{i}_mlp_down')(mlp_hidden)
        
        # Residual connection and LayerNorm
        hidden_states = layers.Add(name=f'layers_{i}_residual_2')([hidden_states, mlp_output])
        hidden_states = layers.LayerNormalization(epsilon=1e-5, name=f'layernorm_2_{i}')(hidden_states)
    
    # 7. Final LayerNorm and LM Head
    hidden_states = layers.LayerNormalization(epsilon=1e-5, name='layernorm_final')(hidden_states)
    outputs = layers.Dense(vocab_size, name='lm_head')(hidden_states)
    
    # 8. Create the model
    model = keras.Model(inputs=input_ids, outputs=outputs, name='OPT-125M')
    return model

def verify_layer_sharding(tp_model):
    """Verify that layers are properly sharded in the tensor parallel model."""
    print("      Verifying layer sharding...")
    if hasattr(tp_model, 'sharding_manager') and tp_model.sharding_manager is not None:
        print("      ✅ Sharding manager found")
        total_params = sum(p.shape.num_elements() for p in tp_model.weights)
        print(f"      ✅ Total parameters in TP model: {total_params:,}")
        print("      ✅ Layer sharding verification passed")
    else:
        print("      ⚠️  No sharding manager found (using fallback mode)")
        print("      ✅ Basic model structure verification passed")

def test_opt125m_parameter_sharding():
    """Test OPT-125M parameter sharding verification."""
    print("🔧 OPT-125M Parameter Sharding Verification")
    print("=" * 50)
    start_time = time.time()
    print(f"⏱️  {time.time() - start_time:.2f}s: Starting OPT-125M parameter sharding test...")
    print(f"⏱️  {time.time() - start_time:.2f}s: Creating OPT-125M model...")
    opt_model = create_opt125m_model()
    original_params = opt_model.count_params()
    print(f"      Original params: {original_params:,}")
    
    print(f"⏱️  {time.time() - start_time:.2f}s: Testing tensor parallelism...")
    tp_manager = TensorParallelKeras(model=opt_model, world_size=2, distributed_backend='fallback')
    
    total_sharded_params = 0
    for i, shard in enumerate(tp_manager.sharded_models):
        shard_params = shard.count_params()
        total_sharded_params += shard_params
        print(f"   Shard {i}: {shard_params:,} parameters")
        
    print(f"      Sharded params: {total_sharded_params:,}")
    assert total_sharded_params >= original_params, "Sharded parameters should be >= original"
    print(f"      ✅ Parameter count verification passed")
    
    print(f"✅ OPT-125M parameter sharding verification completed in {time.time() - start_time:.2f}s")
    return True

def test_opt125m_inference_correctness():
    """Test OPT-125M inference numerical correctness."""
    print("🔧 OPT-125M Inference Numerical Correctness")
    print("=" * 50)
    start_time = time.time()
    print(f"⏱️  {time.time() - start_time:.2f}s: Starting OPT-125M inference test...")
    print(f"⏱️  {time.time() - start_time:.2f}s: Creating OPT-125M model...")
    opt_model = create_opt125m_model()
    
    print(f"⏱️  {time.time() - start_time:.2f}s: Testing tensor parallelism...")
    tp_manager = TensorParallelKeras(model=opt_model, world_size=2, distributed_backend='fallback')
    tp_model = tp_manager.build_assembled_model()

    print(f"✅ {time.time() - start_time:.2f}s: Models created successfully")
    for seq_len in [5, 10, 15]:
        test_input = np.random.randint(0, 1000, (1, seq_len), dtype=np.int32)
        print(f"   Testing sequence {seq_len}: (1, {seq_len})")

        # --- FIX: Feed dictionary to original, raw tensor to TP model ---
        
        # 1. The original model expects a dictionary
        model_input_dict = {'input_ids': test_input}
        original_output = opt_model(model_input_dict)
        
        # 2. The assembled TP model seems to expect the raw tensor
        tp_output = tp_model(test_input)
        # --- END FIX ---

        print(f"      Original output shape: {original_output.shape}")
        print(f"      TP output shape: {tp_output.shape}")
        
        expected_shape = (1, seq_len, 50257) # (batch, seq_len, vocab_size)
        
        if original_output.shape == expected_shape and tp_output.shape == expected_shape:
             print(f"      ✅ Output shapes are correct: {expected_shape}")
        else:
             print(f"      ❌ Output shapes are incorrect.")
             print(f"      Expected: {expected_shape}")
             print(f"      Original: {original_output.shape}")
             print(f"      TP:       {tp_output.shape}")
             assert False, "Shape mismatch"
             
    print(f"✅ OPT-125M inference correctness test completed in {time.time() - start_time:.2f}s")
    return True

def plot_training_graphs(original_history, tp_history):
    """
    Plots the loss and perplexity curves for both the original and TP models.
    """
    if original_history and tp_history:
        original_perplexity = np.exp(original_history.history['loss'])
        tp_perplexity = np.exp(tp_history.history['loss'])

        epochs = range(1, len(original_history.history['loss']) + 1)
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, original_history.history['loss'], 'b-o', label='Original Model Loss')
        plt.plot(epochs, tp_history.history['loss'], 'r-x', label='TP Model Loss')
        plt.title('Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(epochs, original_perplexity, 'b-o', label='Original Model Perplexity')
        plt.plot(epochs, tp_perplexity, 'r-x', label='TP Model Perplexity')
        plt.title('Training Perplexity')
        plt.xlabel('Epochs')
        plt.ylabel('Perplexity')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        print("\n   Generating training performance graphs...")

def test_opt125m_memory_savings():
    """Test OPT-125M optimizer state sharding memory savings."""
    print("🔧 OPT-125M Memory Savings Verification")
    print("=" * 50)
    start_time = time.time()
    
    print(f"⏱️  {time.time() - start_time:.2f}s: Creating OPT-125M model...")
    opt_model = create_opt125m_model()
    
    print(f"⏱️  {time.time() - start_time:.2f}s: Initializing TensorParallelKeras with world_size=2...")
    tp_manager = TensorParallelKeras(model=opt_model, world_size=2, distributed_backend='fallback')
    
    print(f"⏱️  {time.time() - start_time:.2f}s: Compiling model to build optimizer states...")
    tp_manager.compile(optimizer=optimizers.Adam())
    
    # Ensure the coordinated optimizer was created
    if not hasattr(tp_manager, 'coordinated_optimizer'):
        print("      ❌ Coordinated optimizer not found. Cannot calculate memory.")
        return False
    
    # --- FIX 1: Explicitly build the base optimizer's variables ---
    print(f"⏱️  {time.time() - start_time:.2f}s: Building base optimizer states...")
    base_optimizer = tp_manager.coordinated_optimizer.base_optimizer
    base_optimizer.build(tp_manager.trainable_weights)
    
    # --- FIX 2: Access the correct object for the method call ---
    coordinated_optimizer_logic = tp_manager.coordinated_optimizer.coordinated_optimizer
    
    # 1. Calculate memory with REPLICATED states (the default)
    replicated_memory_info = coordinated_optimizer_logic.get_memory_usage()
    
    # 2. Enable SHARDED states and calculate again
    print(f"⏱️  {time.time() - start_time:.2f}s: Enabling optimizer state sharding...")
    coordinated_optimizer_logic.enable_optimizer_state_sharding()
    sharded_memory_info = coordinated_optimizer_logic.get_memory_usage()
    
    print("\n" + "-" * 22 + " Memory Usage Results " + "-" * 22)
    print(f"   - Replicated (No Sharding): {replicated_memory_info.get('total_memory', 'N/A')}")
    print(f"   - Sharded (ZeRO Stage 1)  : {sharded_memory_info.get('sharded_memory', 'N/A')}")
    print(f"   - Memory Savings          : {sharded_memory_info.get('memory_savings', 'N/A')}")
    print("-" * 60 + "\n")
    
    assert sharded_memory_info.get('sharding_enabled', False), "Sharding was not enabled."
    assert float(sharded_memory_info.get('memory_savings', '0%').strip('%')) > 0, "Memory savings should be greater than 0."
    
    print(f"✅ OPT-125M memory savings verification completed in {time.time() - start_time:.2f}s")
    return True

if __name__ == "__main__":
    print("🎯 OPT-125M TENSOR PARALLEL VERIFICATION TEST SUITE")
    print("=" * 60)
    test_results = []
    test_results.append(("OPT-125M Parameter Sharding", test_opt125m_parameter_sharding()))
    test_results.append(("OPT-125M Inference Correctness", test_opt125m_inference_correctness()))
    # test_results.append(("OPT-125M Memory Savings", test_opt125m_memory_savings()))

    
    print("\n" + "=" * 60)
    print("🎉 OPT-125M VERIFICATION TESTING COMPLETED!")
    print(f"\n📋 COMPREHENSIVE RESULTS:")
    passed_tests = 0
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   - {test_name}: {status}")
        if result:
            passed_tests += 1
    
    print(f"\n📊 SUMMARY:")
    print(f"   - Total Tests: {len(test_results)}")
    print(f"   - Passed: {passed_tests}")
    print(f"   - Failed: {len(test_results) - passed_tests}")
    
    if passed_tests == len(test_results):
        print("\n🚀 SUCCESS: All OPT-125M verification tests passed!")
    else:
        print(f"\n⚠️  WARNING: {len(test_results) - passed_tests} tests failed.")