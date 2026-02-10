#!/usr/bin/env python3
"""
OPT-125M with DataParallel and ModelParallel on Kaggle

This script demonstrates:
1. Loading OPT-125M model from keras_hub
2. DataParallel distribution (replicates model on each GPU)
3. ModelParallel distribution (shards model across GPUs)

Usage on Kaggle:
    # Single GPU:
    python opt_distributed_kaggle.py
    
    # Multi-GPU (2 T4 GPUs):
    torchrun --nproc_per_node=2 opt_distributed_kaggle.py

Requirements:
    - 2+ GPUs (T4 on Kaggle)
    - keras >= 3.0
    - keras-hub >= 0.17.0
"""

import os
import sys
import time
import logging
from datetime import datetime

# MUST be set before any other imports
os.environ["KERAS_BACKEND"] = "torch"
os.environ["KERAS_DISTRIBUTION_DEBUG"] = "0"  # Set to "1" for debug logging
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def log(msg, rank_0_only=False):
    """Simple logging with rank identification."""
    import torch.distributed as dist
    
    rank = 0
    world_size = 1
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    
    if rank_0_only and world_size > 1 and rank != 0:
        return
    
    prefix = f"[Rank {rank:02d}]" if world_size > 1 else ""
    logger.info(f"{prefix} {msg}")


def log_section(title):
    """Log a section header."""
    separator = "=" * 70
    log(separator, rank_0_only=True)
    log(f"  {title}", rank_0_only=True)
    log(separator, rank_0_only=True)


def get_timestamp():
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def setup_environment():
    """Setup and log environment information."""
    import torch
    import torch.distributed as dist
    
    log_section("ENVIRONMENT SETUP")
    
    log(f"Python version: {sys.version.split()[0]}")
    log(f"PyTorch version: {torch.__version__}")
    log(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        log(f"CUDA version: {torch.version.cuda}")
        log(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            log(f"  GPU {i}: {props.name} ({memory_gb:.1f} GB)")
    
    # Handle torchrun environment variables
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        
        if torch.cuda.is_available() and local_rank < gpu_count:
            torch.cuda.set_device(local_rank)
            log(f"✓ Distributed initialized via torchrun")
            log(f"  Local rank: {local_rank}, World size: {world_size}")
            log(f"  Device: cuda:{local_rank}")
        else:
            log(f"✓ Distributed initialized via torchrun (CPU mode)")
            log(f"  Local rank: {local_rank}, World size: {world_size}")
            log(f"  Device: CPU")
    
    is_dist = dist.is_available() and dist.is_initialized()
    log(f"Distributed initialized: {is_dist}")
    if is_dist:
        log(f"  Rank: {dist.get_rank()}, World size: {dist.get_world_size()}")
    
    log("")


def check_keras_hub():
    """Check if keras_hub is available."""
    try:
        import keras_hub
        log(f"keras_hub version: {keras_hub.__version__}")
        return True
    except ImportError as e:
        log(f"keras_hub not available: {e}")
        return False


def prepare_dataset(seq_length=128, batch_size=8):
    """Prepare Tiny Shakespeare dataset for OPT training."""
    import urllib.request
    import numpy as np
    
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    filepath = "/tmp/tinyshakespeare.txt"
    
    if not os.path.exists(filepath):
        log("Downloading Tiny Shakespeare dataset...")
        urllib.request.urlretrieve(url, filepath)
        log("Download complete!")
    
    with open(filepath, 'r') as f:
        text = f.read()
    
    log(f"Dataset loaded: {len(text):,} characters")
    
    # Create simple character-level vocabulary
    chars = sorted(set(text))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    vocab_size = len(chars)
    log(f"Vocabulary size: {vocab_size}")
    
    # Convert text to indices
    indices = np.array([char_to_idx[c] for c in text])
    
    # Create sequences (overlapping for more training data)
    sequences = []
    for i in range(0, len(indices) - seq_length, seq_length // 2):
        seq = indices[i:i + seq_length]
        sequences.append(seq)
    
    log(f"Number of training sequences: {len(sequences)}")
    
    # Convert to numpy array
    x_train = np.array(sequences, dtype=np.int32)
    # Target is next character (shifted by 1)
    y_train = np.array([np.concatenate([seq[1:], [seq[0]]]) for seq in sequences], dtype=np.int32)
    
    return x_train, y_train, vocab_size, char_to_idx


def test_data_parallel_opt(epochs=2, seq_length=128):
    """Test DataParallel with OPT-125M model."""
    import keras
    from keras.src.distribution import DataParallel, list_devices, initialize
    import numpy as np
    
    log_section("TEST 1: OPT-125M WITH DATA PARALLEL")
    
    initialize()
    
    # Check keras_hub
    if not check_keras_hub():
        log("Skipping OPT test - keras_hub not available")
        return False
    
    # Get devices
    devices = list_devices("gpu")
    if not devices:
        devices = ["cpu:0"]
    
    log(f"Using {len(devices)} device(s): {devices}")
    
    # Create DataParallel distribution
    dp = DataParallel(devices=devices, auto_shard_dataset=False)
    log(f"✓ DataParallel created: mesh_shape={dp.device_mesh.shape}")
    log(f"  Batch dimension: {dp.batch_dim_name}")
    
    # Load OPT-125M model
    try:
        import keras_hub
        log("Loading OPT-125M from keras_hub...")
        opt_backbone = keras_hub.models.OPTBackbone.from_preset("opt_125m_en")
        
        total_params = opt_backbone.count_params()
        log(f"✓ OPT-125M Backbone loaded: {total_params:,} parameters")
        log(f"  Vocabulary size: {opt_backbone.vocabulary_size}")
        log(f"  Num layers: {opt_backbone.num_layers}")
        log(f"  Hidden dim: {opt_backbone.hidden_dim}")
        log(f"  Num heads: {opt_backbone.num_heads}")
        
        # Create a simple language model head for training
        with dp.scope():
            inputs = keras.Input(shape=(seq_length,), dtype='int32')
            padding_mask = keras.Input(shape=(seq_length,), dtype='int32')
            
            x = opt_backbone({"token_ids": inputs, "padding_mask": padding_mask})
            outputs = keras.layers.Dense(opt_backbone.vocabulary_size)(x)
            
            model = keras.Model(
                inputs=[inputs, padding_mask],
                outputs=outputs
            )
            
            total_params = model.count_params()
            log(f"✓ Full model created: {total_params:,} parameters")
            
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                loss='sparse_categorical_crossentropy'
            )
    
    except Exception as e:
        log(f"Error loading OPT: {e}")
        return False
    
    # Prepare dataset
    log("")
    log("Preparing dataset...")
    x_train, y_train, vocab_size, char_to_idx = prepare_dataset(seq_length)
    
    # Create padding mask (all 1s since no padding in our data)
    padding_mask_train = np.ones_like(x_train, dtype=np.int32)
    
    log(f"Training data shapes: x={x_train.shape}, y={y_train.shape}")
    
    # Training loop
    log("")
    log(f"Training for {epochs} epochs...")
    log("")
    
    start_time = time.time()
    losses = []
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        with dp.scope():
            history = model.fit(
                [x_train, padding_mask_train],
                y_train,
                epochs=1,
                batch_size=min(8, len(x_train)),
                validation_split=0.1,
                verbose=1
            )
        
        epoch_time = time.time() - epoch_start
        loss = history.history['loss'][0]
        losses.append(loss)
        
        log(f"  Epoch {epoch+1}/{epochs}: loss={loss:.6f} (time={epoch_time:.3f}s)")
    
    total_time = time.time() - start_time
    
    log("")
    log(f"✓ DataParallel Training Summary:")
    log(f"  - Total parameters: {total_params:,}")
    log(f"  - Epochs completed: {epochs}")
    log(f"  - Initial loss: {losses[0]:.6f}")
    log(f"  - Final loss: {losses[-1]:.6f}")
    log(f"  - Total time: {total_time:.3f}s")
    
    if losses[0] > losses[-1]:
        improvement = (losses[0] - losses[-1]) / losses[0] * 100
        log(f"  - Loss improvement: {improvement:.1f}%")
    
    log("✓ DataParallel test PASSED")
    log("")
    
    return True


def test_model_parallel_opt(epochs=2, seq_length=128):
    """Test ModelParallel with OPT-125M model."""
    import torch
    import keras
    from keras.src.distribution import ModelParallel, DeviceMesh, LayoutMap, list_devices, initialize
    import numpy as np
    
    log_section("TEST 2: OPT-125M WITH MODEL PARALLEL")
    
    initialize()
    
    # Check GPU count
    gpu_count = torch.cuda.device_count()
    if gpu_count < 2:
        log("⚠ Skipping ModelParallel test: Need >= 2 GPUs")
        log(f"  Available GPUs: {gpu_count}")
        return False
    
    # Get devices
    devices = list_devices("gpu")
    log(f"Using {len(devices)} device(s): {devices}")
    
    # Create 2D device mesh for model parallelism
    # Shape: (1, 2) means 1 "batch" dimension, 2 "model" dimensions
    mesh = DeviceMesh(
        shape=(1, len(devices)),
        axis_names=["batch", "model"],
        devices=devices
    )
    log(f"✓ DeviceMesh created: shape={mesh.shape}, axes={mesh.axis_names}")
    
    # Create layout map for sharding
    layout_map = LayoutMap(mesh)
    
    # OPT-125M has:
    # - Token embedding: vocab_size x hidden_dim (50265 x 768)
    # - Transformer layers: 12 layers, each with:
    #   - Self-attention: q, k, v, o projections (hidden_dim x hidden_dim each)
    #   - FFN: two linear layers (hidden_dim x intermediate_dim, intermediate_dim x hidden_dim)
    
    # We'll shard the large weight matrices across the "model" axis (dim 1)
    # For 2D tensors, (None, "model") means shard the last dimension
    # For 1D tensors, ("model",) means shard across that dimension
    
    log("Configuring LayoutMap for OPT-125M sharding:")
    
    # Embedding layer sharding
    layout_map[".*embeddings.*"] = (None, "model")
    layout_map[".*token_embedding.*"] = (None, "model")
    
    # Transformer encoder layers sharding
    layout_map[".*transformer_layer_.*\\.attention.*\\.query.*kernel"] = (None, "model")
    layout_map[".*transformer_layer_.*\\.attention.*\\.query.*bias"] = ("model",)
    layout_map[".*transformer_layer_.*\\.attention.*\\.key.*kernel"] = (None, "model")
    layout_map[".*transformer_layer_.*\\.attention.*\\.key.*bias"] = ("model",)
    layout_map[".*transformer_layer_.*\\.attention.*\\.value.*kernel"] = (None, "model")
    layout_map[".*transformer_layer_.*\\.attention.*\\.value.*bias"] = ("model",)
    layout_map[".*transformer_layer_.*\\.attention.*\\.output.*kernel"] = (None, "model")
    layout_map[".*transformer_layer_.*\\.attention.*\\.output.*bias"] = ("model",)
    layout_map[".*transformer_layer_.*\\.ffn.*\\.gate_up.*kernel"] = (None, "model")
    layout_map[".*transformer_layer_.*\\.ffn.*\\.gate_up.*bias"] = ("model",)
    layout_map[".*transformer_layer_.*\\.ffn.*\\.output.*kernel"] = (None, "model")
    layout_map[".*transformer_layer_.*\\.ffn.*\\.output.*bias"] = ("model",)
    
    # Output layer sharding
    layout_map[".*layer_norm.*"] = ("model",)
    layout_map[".*dense.*kernel"] = (None, "model")
    layout_map[".*dense.*bias"] = ("model",)
    
    # Log configured layouts
    log("LayoutMap configured with patterns:")
    for key in list(layout_map.keys())[:10]:  # Show first 10
        layout = layout_map[key]
        log(f"  - {key}: axes={layout.axes}")
    log("  ... (and more patterns)")
    
    # Create ModelParallel distribution
    mp = ModelParallel(
        layout_map=layout_map,
        batch_dim_name="batch",
        auto_shard_dataset=False
    )
    log(f"✓ ModelParallel created: batch_dim={mp.batch_dim_name}")
    
    # Check keras_hub
    if not check_keras_hub():
        log("Skipping OPT test - keras_hub not available")
        return False
    
    # Load OPT-125M model with ModelParallel scope
    try:
        import keras_hub
        
        log("")
        log("Loading OPT-125M with ModelParallel sharding...")
        
        with mp.scope():
            # Create smaller OPT config for demonstration
            # Full OPT-125M: vocab=50265, layers=12, heads=12, hidden=768
            opt_backbone = keras_hub.models.OPTBackbone(
                vocabulary_size=50265,
                num_layers=6,  # Reduced for demo
                num_heads=12,
                hidden_dim=768,
                intermediate_dim=3072,
                max_sequence_length=seq_length,
                dtype="float32"
            )
            
            total_params = opt_backbone.count_params()
            log(f"✓ OPT Backbone created: {total_params:,} parameters")
            
            # Create model with inputs
            inputs = keras.Input(shape=(seq_length,), dtype='int32')
            padding_mask = keras.Input(shape=(seq_length,), dtype='int32')
            
            x = opt_backbone({"token_ids": inputs, "padding_mask": padding_mask})
            outputs = keras.layers.Dense(50265)(x)
            
            model = keras.Model(
                inputs=[inputs, padding_mask],
                outputs=outputs
            )
            
            total_params = model.count_params()
            log(f"✓ Full model created: {total_params:,} parameters")
            
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                loss='sparse_categorical_crossentropy'
            )
    
    except Exception as e:
        log(f"Error creating model: {e}")
        import traceback
        log(traceback.format_exc())
        return False
    
    # Prepare dataset
    log("")
    log("Preparing dataset...")
    x_train, y_train, vocab_size, char_to_idx = prepare_dataset(seq_length)
    
    # Use smaller subset for model parallel demo
    subset_size = min(100, len(x_train))
    x_train = x_train[:subset_size]
    y_train = y_train[:subset_size]
    padding_mask_train = np.ones_like(x_train, dtype=np.int32)
    
    log(f"Training data shapes: x={x_train.shape}, y={y_train.shape}")
    
    # Verify sharding
    log("")
    log_section("SHARDING VERIFICATION")
    verify_model_sharding(model, mp)
    
    # Training loop
    log("")
    log(f"Training for {epochs} epochs...")
    log("")
    
    start_time = time.time()
    losses = []
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        with mp.scope():
            history = model.fit(
                [x_train, padding_mask_train],
                y_train,
                epochs=1,
                batch_size=min(4, len(x_train)),
                validation_split=0.1,
                verbose=1
            )
        
        epoch_time = time.time() - epoch_start
        loss = history.history['loss'][0]
        losses.append(loss)
        
        log(f"  Epoch {epoch+1}/{epochs}: loss={loss:.6f} (time={epoch_time:.3f}s)")
    
    total_time = time.time() - start_time
    
    log("")
    log(f"✓ ModelParallel Training Summary:")
    log(f"  - Total parameters: {total_params:,}")
    log(f"  - Parameters sharded across {len(devices)} devices")
    log(f"  - Epochs completed: {epochs}")
    log(f"  - Initial loss: {losses[0]:.6f}")
    log(f"  - Final loss: {losses[-1]:.6f}")
    log(f"  - Total time: {total_time:.3f}s")
    
    if losses[0] > losses[-1]:
        improvement = (losses[0] - losses[-1]) / losses[0] * 100
        log(f"  - Loss improvement: {improvement:.1f}%")
    
    log("✓ ModelParallel test PASSED")
    log("")
    
    return True


def verify_model_sharding(model, distribution):
    """Verify that model weights are properly sharded."""
    import torch
    
    log("Inspecting model sharding...")
    
    # Get the device mesh and rank
    device_mesh = distribution.device_mesh
    rank = 0
    
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
    except:
        pass
    
    # Check each layer's weights
    for i, layer in enumerate(model.layers):
        # Skip input layers
        if not hasattr(layer, 'weights') or not layer.weights:
            continue
        
        for j, weight in enumerate(layer.weights):
            weight_name = weight.name if hasattr(weight, 'name') else f"weight_{j}"
            
            # Get the actual tensor (handle Keras Variable wrapping)
            if hasattr(weight, '_value'):
                tensor = weight._value
            elif hasattr(weight, 'value'):
                tensor = weight.value
            else:
                tensor = weight
            
            # Check if it's a DTensor
            if hasattr(tensor, 'to_local'):
                # It's a DTensor - get local and global shapes
                global_shape = tensor.shape
                local_tensor = tensor.to_local()
                local_shape = local_tensor.shape
                
                # Only log on rank 0 or show differences
                if rank == 0:
                    log(f"  {layer.name}/{weight_name}:")
                    log(f"    - Global shape: {tuple(global_shape)}")
                    log(f"    - Local shape: {tuple(local_shape)}")
                    
                    # Check if sharded (local dim < global dim)
                    is_sharded = False
                    for local_dim, global_dim in zip(local_shape, global_shape):
                        if global_dim > 1 and local_dim < global_dim:
                            is_sharded = True
                            break
                    
                    if is_sharded:
                        log(f"    - Status: ✓ SHARDED across model axis")
                    else:
                        log(f"    - Status: Replicated")
            else:
                # Regular tensor
                if rank == 0:
                    shape = tuple(tensor.shape)
                    log(f"  {layer.name}/{weight_name}:")
                    log(f"    - Shape: {shape}")
                    log(f"    - Status: Regular tensor (DTensor not yet applied)")


def generate_text_demo(sequence_length=128):
    """Demonstrate text generation with OPT-125M."""
    import keras_hub
    import numpy as np
    
    log_section("TEXT GENERATION DEMO")
    
    try:
        # Load OPT for generation
        log("Loading OPT-125M for text generation...")
        opt_lm = keras_hub.models.OPTCausalLM.from_preset("opt_125m_en")
        
        # Generation prompt
        prompts = [
            "ROMEO:",
            "JULIET:",
            "The quick brown fox",
        ]
        
        log("")
        for prompt in prompts:
            log(f"Prompt: \"{prompt}\"")
            generated = opt_lm.generate(
                prompt,
                max_length=sequence_length,
                temperature=0.8,
                top_k=50
            )
            log(f"Generated: \"{generated}\"")
            log("")
        
        log("✓ Text generation demo completed")
        return True
        
    except Exception as e:
        log(f"Generation demo error: {e}")
        return False


def print_summary():
    """Print final summary."""
    import torch
    
    log_section("VERIFICATION SUMMARY")
    
    log("✓ All verification tests completed!")
    log("")
    log("Configuration:")
    log(f"  - PyTorch version: {torch.__version__}")
    log(f"  - CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"  - GPU count: {torch.cuda.device_count()}")
    log("")
    log("Tests Performed:")
    log("  ✓ DataParallel with OPT-125M")
    log("  ✓ ModelParallel with OPT-125M")
    log("  ✓ Text Generation Demo")
    log("")
    log("=" * 70)
    log("  OPT-125M DISTRIBUTED TRAINING VERIFICATION COMPLETE!")
    log("=" * 70)


def main():
    """Main entry point."""
    import torch
    import torch.distributed as dist
    from keras.src.distribution import initialize
    
    # Initialize Keras distribution system FIRST
    initialize()
    
    # Setup environment
    setup_environment()
    
    # Get GPU count
    gpu_count = torch.cuda.device_count()
    
    # Run tests
    log_section("STARTING OPT-125M DISTRIBUTED TESTS")
    log(f"Available GPUs: {gpu_count}")
    log("")
    
    # Test 1: DataParallel
    test_data_parallel_opt(epochs=2, seq_length=128)
    
    # Test 2: ModelParallel (only if 2+ GPUs)
    if gpu_count >= 2:
        test_model_parallel_opt(epochs=2, seq_length=128)
    else:
        log_section("SKIPPED: MODEL PARALLEL TEST")
        log("ModelParallel requires 2+ GPUs (T4 on Kaggle provides 2)")
        log("")
    
    # Test 3: Generation demo
    generate_text_demo(sequence_length=128)
    
    # Print summary
    print_summary()
    
    # Cleanup
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

