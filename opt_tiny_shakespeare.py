#!/usr/bin/env python3
"""
OPT-125M Training on Tiny Shakespeare Dataset with Distributed Training

This script trains an OPT-125M style transformer model on Tiny Shakespeare
dataset with DataParallel and ModelParallel distributions.

Usage:
    python opt_tiny_shakespeare.py
    
    # Multi-GPU:
    torchrun --nproc_per_node=2 opt_tiny_shakespeare.py
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
    import os
    
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


def prepare_text_dataset(text, seq_length=128):
    """Prepare character-level text dataset."""
    import numpy as np
    
    chars = sorted(set(text))
    vocab_size = len(chars)
    log(f"Vocabulary size: {vocab_size}")
    
    char_to_idx = {c: i for i, c in enumerate(chars)}
    indices = np.array([char_to_idx[c] for c in text])
    
    sequences = []
    targets = []
    
    for i in range(0, len(indices) - seq_length, seq_length // 2):
        seq = indices[i:i + seq_length]
        target = indices[i + 1:i + seq_length + 1]
        sequences.append(seq)
        targets.append(target)
    
    log(f"Number of training sequences: {len(sequences)}")
    
    return sequences, targets, vocab_size, char_to_idx, idx_to_char


def create_opt_model(
    vocab_size: int,
    seq_length: int = 128,
    hidden_dim: int = 768,
    num_layers: int = 12,
    num_heads: int = 12,
    intermediate_dim: int = 3072
):
    """
    Create OPT-style transformer model.
    
    OPT-125M Architecture:
    - hidden_dim: 768
    - intermediate_dim: 3072 (4x hidden_dim)
    - num_layers: 12
    - num_heads: 12
    - vocab_size: 50264 (BPE) or 65 (char-level)
    
    Parameter count for 125M:
    - With char vocab (65): ~85M parameters
    - With BPE vocab (50264): ~125M parameters
    """
    import keras
    from keras import layers
    import numpy as np
    
    # Input layer
    inputs = layers.Input(shape=(seq_length,), dtype='int32', name='input_ids')
    
    # Token embedding
    x = layers.Embedding(
        input_dim=vocab_size,
        output_dim=hidden_dim,
        name='token_embedding'
    )(inputs)
    
    # Positional encoding (learned)
    pos_emb = layers.Embedding(
        input_dim=seq_length,
        output_dim=hidden_dim,
        name='position_embedding'
    )(layers.Range(start=0, dtype='int32', limit=seq_length))
    x = x + pos_emb
    
    # Transformer blocks
    for i in range(num_layers):
        # Multi-head self-attention
        attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=hidden_dim // num_heads,
            name=f'attention_{i}'
        )(x, x)
        
        attn = layers.Dropout(0.1)(attn)
        x = layers.Add()([x, attn])
        x = layers.LayerNormalization(epsilon=1e-5)(x)
        
        # Feed-forward network
        ffn = keras.Sequential([
            layers.Dense(intermediate_dim, activation='gelu', name=f'ffn_dense1_{i}'),
            layers.Dense(hidden_dim, name=f'ffn_dense2_{i}')
        ], name=f'ffn_{i}')
        
        ffn_out = ffn(x)
        ffn_out = layers.Dropout(0.1)(ffn_out)
        
        x = layers.Add()([x, ffn_out])
        x = layers.LayerNormalization(epsilon=1e-5)(x)
    
    # Output layer
    x = layers.Dense(hidden_dim, activation='gelu', name='output_dense')(x)
    outputs = layers.Dense(vocab_size, name='logits')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='OPT_model')
    
    return model


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


def train_data_parallel(epochs=3, seq_length=128):
    """Train OPT model with DataParallel."""
    log_section("OPT-125M DATA PARALLEL TRAINING")
    
    import keras
    from keras.distribution import DataParallel, list_devices
    import numpy as np
    
    devices = list_devices("gpu")
    if not devices:
        devices = ["cpu:0"]
    
    log(f"Using devices: {devices}")
    
    # Load dataset
    text = download_tiny_shakespeare()
    sequences, targets, vocab_size, _, _, _ = prepare_text_dataset(text, seq_length)
    x_train = np.array(sequences, dtype=np.int32)
    y_train = np.array(targets, dtype=np.int32)
    
    log(f"Data shapes: x={x_train.shape}, y={y_train.shape}")
    
    # Create distribution
    dp = DataParallel(devices=devices, auto_shard_dataset=False)
    log(f"DataParallel: mesh_shape={dp.device_mesh.shape}")
    
    # OPT-125M architecture
    hidden_dim = 768
    num_layers = 12
    num_heads = 12
    intermediate_dim = 3072
    
    log(f"")
    log(f"OPT-125M Model Configuration:")
    log(f"  - hidden_dim: {hidden_dim}")
    log(f"  - intermediate_dim: {intermediate_dim}")
    log(f"  - num_layers: {num_layers}")
    log(f"  - num_heads: {num_heads}")
    log(f"  - vocab_size: {vocab_size} (character-level)")
    
    with dp.scope():
        model = create_opt_model(
            vocab_size=vocab_size,
            seq_length=seq_length,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            intermediate_dim=intermediate_dim
        )
        
        total_params = model.count_params()
        log(f"")
        log(f"✓ Model created with {total_params:,} parameters")
        
        # Parameter breakdown
        log(f"")
        log(f"Parameter Breakdown:")
        log(f"  - Token embedding: {vocab_size * hidden_dim:,}")
        log(f"  - Position embedding: {seq_length * hidden_dim:,}")
        log(f"  - Per layer (attention): {3 * hidden_dim * hidden_dim + hidden_dim:,}")
        log(f"  - Per layer (FFN): {2 * hidden_dim * intermediate_dim + intermediate_dim:,}")
        log(f"  - {num_layers} layers: ~{(3 * hidden_dim * hidden_dim + hidden_dim + 2 * hidden_dim * intermediate_dim + intermediate_dim) * num_layers:,}")
        log(f"  - Output layer: {hidden_dim * vocab_size:,}")
        log(f"")
        log(f"  Note: Full OPT-125M uses BPE vocab (50264), giving ~125M params")
        log(f"  Character-level (65) gives ~85M params")
        log(f"  To match 125M, add more layers or use BPE tokenization")
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    log(f"")
    log(f"Training for {epochs} epochs...")
    log(f"")
    
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
        
        log(f"  Epoch {epoch+1}/{epochs}: "
            f"loss={history.history['loss'][0]:.4f}, "
            f"val_loss={history.history['val_loss'][0]:.4f}")
    
    log(f"")
    log(f"✓ DataParallel Complete: {total_params:,} params, "
        f"final_loss={history.history['loss'][-1]:.4f}")
    log(f"")
    
    return True


def train_model_parallel(epochs=3, seq_length=128):
    """Train OPT model with ModelParallel."""
    log_section("OPT-125M MODEL PARALLEL TRAINING")
    
    import torch
    from keras.distribution import ModelParallel, DeviceMesh, LayoutMap, list_devices
    
    if torch.cuda.device_count() < 2:
        log("Skipping: Need >= 2 GPUs for ModelParallel")
        return False
    
    devices = list_devices("gpu")
    log(f"Using devices: {devices}")
    
    # Load dataset
    text = download_tiny_shakespeare()
    sequences, targets, vocab_size, _, _, _ = prepare_text_dataset(text, seq_length)
    import numpy as np
    x_train = np.array(sequences, dtype=np.int32)
    y_train = np.array(targets, dtype=np.int32)
    
    # Create device mesh
    mesh = DeviceMesh(
        shape=(1, len(devices)),
        axis_names=["batch", "model"],
        devices=devices
    )
    log(f"DeviceMesh: shape={mesh.shape}")
    
    # Create layout map for weight sharding
    layout_map = LayoutMap(mesh)
    layout_map[".*token_embedding"] = (None, "model")
    layout_map[".*position_embedding"] = (None, "model")
    layout_map[".*attention.*kernel"] = (None, "model")
    layout_map[".*ffn.*kernel"] = (None, "model")
    layout_map[".*output_dense.*kernel"] = (None, "model")
    layout_map[".*bias"] = ("model",)
    
    log("LayoutMap configured for weight sharding")
    
    # Create distribution
    mp = ModelParallel(
        layout_map=layout_map,
        batch_dim_name="batch",
        auto_shard_dataset=False
    )
    log(f"ModelParallel: batch_dim={mp.batch_dim_name}")
    
    # Larger model for sharding demo
    hidden_dim = 1024
    num_layers = 12
    num_heads = 16
    intermediate_dim = 4096
    
    log(f"")
    log(f"OPT-style Model with Weight Sharding:")
    log(f"  - hidden_dim: {hidden_dim}")
    log(f"  - num_layers: {num_layers}")
    log(f"  - Sharding: weights split across {len(devices)} GPUs")
    
    with mp.scope():
        import keras
        model = create_opt_model(
            vocab_size=vocab_size,
            seq_length=seq_length,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            intermediate_dim=intermediate_dim
        )
        
        total_params = model.count_params()
        log(f"✓ Model: {total_params:,} parameters (sharded)")
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    log(f"")
    log(f"Training for {epochs} epochs...")
    log(f"")
    
    for epoch in range(epochs):
        with mp.scope():
            history = model.fit(
                x_train, y_train,
                epochs=1,
                batch_size=32,
                validation_split=0.1,
                verbose=1
            )
        
        log(f"  Epoch {epoch+1}/{epochs}: loss={history.history['loss'][0]:.4f}")
    
    log(f"")
    log(f"✓ ModelParallel Complete: {total_params:,} params (sharded)")
    log(f"")
    
    return True


def main():
    """Main entry point."""
    import torch
    import torch.distributed as dist
    from keras.distribution import initialize
    
    setup_environment()
    initialize()
    
    # DataParallel (always runs)
    train_data_parallel(epochs=3)
    
    # ModelParallel (if 2+ GPUs)
    if torch.cuda.device_count() >= 2:
        train_model_parallel(epochs=3)
    else:
        log_section("MODEL PARALLEL SKIPPED")
        log("Need 2+ GPUs")
        log("")
    
    if dist.is_initialized():
        dist.destroy_process_group()
    
    log_section("COMPLETE")
    log("OPT-125M distributed training verification done!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

