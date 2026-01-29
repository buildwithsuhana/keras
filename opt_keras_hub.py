#!/usr/bin/env python3
"""
OPT-125M Training from keras_hub on Tiny Shakespeare Dataset

This script demonstrates:
1. Loading OPT-125M model from keras_hub
2. Fine-tuning on Tiny Shakespeare dataset
3. DataParallel and ModelParallel distributions

Usage:
    python opt_keras_hub.py
    
    # Multi-GPU:
    torchrun --nproc_per_node=2 opt_keras_hub.py
"""

import os
os.environ["KERAS_BACKEND"] = "torch"
os.environ["KERAS_DISTRIBUTION_DEBUG"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import time
import logging

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


def download_tiny_shakespeare():
    """Download Tiny Shakespeare dataset."""
    import urllib.request
    
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    filepath = "/tmp/tinyshakespeare.txt"
    
    if not os.path.exists(filepath):
        log("Downloading Tiny Shakespeare dataset...")
        urllib.request.urlretrieve(url, filepath)
        log("Download complete!")
    
    with open(filepath, 'r') as f:
        text = f.read()
    
    log(f"Dataset loaded: {len(text):,} characters")
    return text


def prepare_dataset(text, seq_length=512, batch_size=8):
    """Prepare dataset for OPT model."""
    import numpy as np
    
    # OPT uses BPE tokenization, but for fine-tuning we'll use character-level
    # and map to the OPT vocab (or use a simple tokenizer)
    chars = sorted(set(text))
    vocab_size = len(chars)
    log(f"Vocabulary size: {vocab_size}")
    
    # For keras_hub OPT, we need to use their tokenizer
    # But since we're fine-tuning, we'll create input IDs manually
    char_to_idx = {c: i for i, c in enumerate(chars)}
    indices = np.array([char_to_idx[c] for c in text])
    
    # Create sequences
    sequences = []
    for i in range(0, len(indices) - seq_length, seq_length // 2):
        seq = indices[i:i + seq_length]
        sequences.append(seq)
    
    log(f"Number of training sequences: {len(sequences)}")
    
    return sequences, vocab_size, char_to_idx


def setup_environment():
    """Setup environment."""
    import torch
    
    log_section("ENVIRONMENT SETUP")
    
    log(f"Python version: {sys.version.split()[0]}")
    log(f"PyTorch version: {torch.__version__}")
    log(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        log(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            log(f"  GPU {i}: {props.name}")
    
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        log(f"Distributed: rank={local_rank}, world_size={world_size}")
    
    log("")


def check_keras_hub():
    """Check if keras_hub is available and import OPT."""
    try:
        import keras_hub
        log(f"keras_hub version: {keras_hub.__version__}")
        return True
    except ImportError as e:
        log(f"keras_hub not available: {e}")
        log("Install with: pip install keras-hub")
        return False


def train_with_keras_hub_opt(epochs=1, seq_length=512):
    """Train using keras_hub OPT model."""
    log_section("OPT-125M FROM KERAS_HUB")
    
    # Check keras_hub
    if not check_keras_hub():
        return False
    
    import numpy as np
    import keras
    from keras.src.distribution import DataParallel, list_devices
    
    # Try to load OPT from keras_hub
    try:
        import keras_hub
        
        # Load OPT-125M pre-trained model
        log("Loading OPT-125M from keras_hub...")
        opt_model = keras_hub.models.OptCausalLM.from_preset("opt_125m_en")
        
        total_params = opt_model.count_params()
        log(f"✓ OPT-125M loaded: {total_params:,} parameters")
        log(f"  Model: {opt_model.name}")
        
        # Show model architecture summary
        log(f"")
        log(f"Model Configuration:")
        log(f"  - Vocabulary size: {opt_model.vocabulary_size}")
        log(f"  - Sequence length: {opt_model.max_sequence_length}")
        
    except Exception as e:
        log(f"Could not load keras_hub OPT: {e}")
        log("Using custom OPT model instead...")
        return train_custom_opt(epochs, seq_length)
    
    # Get devices
    devices = list_devices("gpu")
    if not devices:
        devices = ["cpu:0"]
    
    log(f"")
    log(f"Using devices: {devices}")
    
    # Create distribution
    dp = DataParallel(devices=devices, auto_shard_dataset=False)
    log(f"DataParallel: mesh_shape={dp.device_mesh.shape}")
    
    # Load dataset
    text = download_tiny_shakespeare()
    sequences, vocab_size, char_to_idx = prepare_dataset(text, seq_length)
    
    # Since keras_hub OPT has its own tokenizer, we need to use it
    # For this demo, we'll use a simple approach with the pre-trained model
    log(f"")
    log(f"Training configuration:")
    log(f"  - Epochs: {epochs}")
    log(f"  - Sequence length: {seq_length}")
    log(f"  - Batch size: 8")
    log(f"")
    
    # For fine-tuning, we need to prepare data for the OPT model
    # OPT expects tokenized input, so we use the model's tokenizer
    try:
        # Tokenize a small subset for demonstration
        sample_text = text[:10000]  # Use first 10K chars for demo
        log(f"Tokenizing sample text ({len(sample_text)} chars)...")
        
        # Use keras_hub's tokenizer
        tokenizer = keras_hub.models.OptTokenizer.from_preset("opt_125m_en")
        tokenized = tokenizer(sample_text)
        
        log(f"Tokenized shape: {tokenized.shape}")
        log(f"Vocabulary size: {tokenizer.vocabulary_size}")
        
        # Prepare training data
        x = tokenized[:, :-1]  # Input (all but last token)
        y = tokenized[:, 1:]   # Target (all but first token)
        
        log(f"Training data: x={x.shape}, y={y.shape}")
        
        # Training loop
        log(f"")
        log(f"Starting fine-tuning...")
        log(f"")
        
        start_time = time.time()
        
        # Use smaller subset for demo
        subset_size = min(100, x.shape[0])
        x_subset = x[:subset_size]
        y_subset = y[:subset_size]
        
        for epoch in range(epochs):
            with dp.scope():
                history = opt_model.fit(
                    x_subset, y_subset,
                    epochs=1,
                    batch_size=2,
                    validation_split=0.1,
                    verbose=1
                )
            
            log(f"  Epoch {epoch+1}/{epochs}: "
                f"loss={history.history['loss'][0]:.4f}")
        
        total_time = time.time() - start_time
        
        log(f"")
        log(f"✓ Training Complete!")
        log(f"  - Parameters: {total_params:,}")
        log(f"  - Final loss: {history.history['loss'][-1]:.4f}")
        log(f"  - Time: {total_time:.1f}s")
        
        # Generate sample text
        log(f"")
        log("Generating sample text...")
        start_prompt = "ROMEO:"
        
        # Tokenize prompt
        prompt_tokens = tokenizer(start_prompt)
        
        # Generate
        generated = opt_model.generate(
            prompt_tokens,
            max_length=100,
            temperature=0.8
        )
        
        # Decode
        generated_text = tokenizer.detokenize(generated)
        log(f"")
        log("=" * 70)
        log("GENERATED TEXT:")
        log("=" * 70)
        log(generated_text)
        log("=" * 70)
        
    except Exception as e:
        log(f"Training error: {e}")
        log("Falling back to custom OPT model...")
        return train_custom_opt(epochs, seq_length)
    
    return True


def train_custom_opt(epochs=3, seq_length=128):
    """Train custom OPT-style model if keras_hub not available."""
    log_section("CUSTOM OPT MODEL")
    
    import numpy as np
    import keras
    from keras import layers
    from keras.src.distribution import DataParallel, list_devices
    
    devices = list_devices("gpu")
    if not devices:
        devices = ["cpu:0"]
    
    log(f"Using devices: {devices}")
    
    # Load dataset
    text = download_tiny_shakespeare()
    sequences, vocab_size, char_to_idx = prepare_dataset(text, seq_length)
    x_train = np.array(sequences, dtype=np.int32)
    y_train = np.array([np.concatenate([seq[1:], [seq[0]]]) for seq in sequences], dtype=np.int32)
    
    # Create distribution
    dp = DataParallel(devices=devices, auto_shard_dataset=False)
    log(f"DataParallel: mesh_shape={dp.device_mesh.shape}")
    
    # OPT-125M configuration
    hidden_dim = 768
    num_layers = 12
    num_heads = 12
    intermediate_dim = 3072
    
    log(f"")
    log(f"Custom OPT Model Configuration:")
    log(f"  - hidden_dim: {hidden_dim}")
    log(f"  - num_layers: {num_layers}")
    log(f"  - num_heads: {num_heads}")
    log(f"  - vocab_size: {vocab_size}")
    
    # Create model
    with dp.scope():
        # Input
        inputs = layers.Input(shape=(seq_length,), dtype='int32')
        
        # Token embedding
        x = layers.Embedding(vocab_size, hidden_dim)(inputs)
        
        # Position embedding
        positions = layers.Embedding(seq_length, hidden_dim)(
            layers.Range(start=0, dtype='int32', limit=seq_length)
        )
        x = x + positions
        
        # Transformer blocks
        for i in range(num_layers):
            # Multi-head attention
            attn = layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim=hidden_dim // num_heads
            )(x, x)
            x = layers.Dropout(0.1)(x)
            x = layers.Add()([x, layers.LayerNormalization()(x)])
            
            # FFN
            ffn = keras.Sequential([
                layers.Dense(intermediate_dim, activation='gelu'),
                layers.Dense(hidden_dim)
            ])(x)
            x = layers.Dropout(0.1)(ffn)
            x = layers.Add()([x, layers.LayerNormalization()(x)])
        
        # Output
        outputs = layers.Dense(vocab_size)(x)
        
        model = keras.Model(inputs, outputs)
        
        total_params = model.count_params()
        log(f"✓ Model created: {total_params:,} parameters")
        
        model.compile(
            optimizer=keras.optimizers.Adam(0.0001),
            loss='sparse_categorical_crossentropy'
        )
    
    log(f"")
    log(f"Training for {epochs} epochs...")
    
    start_time = time.time()
    
    for epoch in range(epochs):
        with dp.scope():
            history = model.fit(
                x_train, y_train,
                epochs=1,
                batch_size=32,
                validation_split=0.1,
                verbose=1
            )
        
        log(f"  Epoch {epoch+1}/{epochs}: loss={history.history['loss'][0]:.4f}")
    
    log(f"")
    log(f"✓ Complete: {total_params:,} params, loss={history.history['loss'][-1]:.4f}")
    
    return True


def train_model_parallel(epochs=3, seq_length=128):
    """Train with ModelParallel distribution."""
    log_section("OPT MODEL PARALLEL")
    
    import torch
    from keras.src.distribution import ModelParallel, DeviceMesh, LayoutMap, list_devices
    
    if torch.cuda.device_count() < 2:
        log("Skipping: Need 2+ GPUs")
        return False
    
    devices = list_devices("gpu")
    log(f"Using devices: {devices}")
    
    # Create device mesh
    mesh = DeviceMesh(
        shape=(1, len(devices)),
        axis_names=["batch", "model"],
        devices=devices
    )
    
    layout_map = LayoutMap(mesh)
    layout_map[".*embedding"] = (None, "model")
    layout_map[".*kernel"] = (None, "model")
    layout_map[".*bias"] = ("model",)
    
    mp = ModelParallel(
        layout_map=layout_map,
        batch_dim_name="batch",
        auto_shard_dataset=False
    )
    
    log(f"ModelParallel: {mp.batch_dim_name}")
    
    # Use larger model for sharding demo
    import keras
    from keras import layers
    import numpy as np
    
    text = download_tiny_shakespeare()
    sequences, vocab_size, _ = prepare_dataset(text, seq_length)
    x_train = np.array(sequences[:100], dtype=np.int32)
    
    hidden_dim = 1024
    num_layers = 12
    
    with mp.scope():
        inputs = layers.Input(shape=(seq_length,))
        x = layers.Embedding(vocab_size, hidden_dim)(inputs)
        
        for i in range(num_layers):
            attn = layers.MultiHeadAttention(16, 64)(x, x)
            x = layers.Add()([x, layers.LayerNormalization()(attn)])
            ffn = keras.Sequential([
                layers.Dense(4096, activation='gelu'),
                layers.Dense(hidden_dim)
            ])(x)
            x = layers.Add()([x, layers.LayerNormalization()(ffn)])
        
        outputs = layers.Dense(vocab_size)(x)
        model = keras.Model(inputs, outputs)
        
        total_params = model.count_params()
        log(f"✓ Model: {total_params:,} parameters (sharded)")
        
        model.compile(
            optimizer=keras.optimizers.Adam(0.0001),
            loss='sparse_categorical_crossentropy'
        )
    
    log(f"Training...")
    for epoch in range(epochs):
        with mp.scope():
            history = model.fit(x_train, x_train, epochs=1, verbose=1)
        log(f"  Epoch {epoch+1}/{epochs}: loss={history.history['loss'][0]:.4f}")
    
    log(f"✓ ModelParallel complete: {total_params:,} params")
    
    return True


def main():
    """Main entry point."""
    import torch
    import torch.distributed as dist
    from keras.src.distribution import initialize
    
    setup_environment()
    initialize()
    
    # Try keras_hub OPT first
    success = train_with_keras_hub_opt(epochs=1)
    
    if not success:
        # Fall back to custom model
        train_custom_opt(epochs=2)
    
    # ModelParallel (if 2+ GPUs)
    if torch.cuda.device_count() >= 2:
        train_model_parallel(epochs=2)
    
    if dist.is_initialized():
        dist.destroy_process_group()
    
    log_section("COMPLETE")
    log("OPT distributed training verification done!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

