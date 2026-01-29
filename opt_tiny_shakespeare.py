#!/usr/bin/env python3
"""
OPT-125M Training on Tiny Shakespeare Dataset with Distributed Training

This script demonstrates:
1. Loading Tiny Shakespeare dataset
2. Creating an OPT-style transformer model
3. Training with DataParallel and ModelParallel distributions

Usage:
    python opt_tiny_shakespeare.py
    
    # Multi-GPU:
    torchrun --nproc_per_node=2 opt_tiny_shakespeare.py
"""

import os
# MUST be set before any other imports
os.environ["KERAS_BACKEND"] = "torch"
os.environ["KERAS_DISTRIBUTION_DEBUG"] = "0"  # Set to 1 for debug logs
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import time
import logging

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


def download_tiny_shakespeare():
    """Download and prepare Tiny Shakespeare dataset."""
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
    """Prepare character-level text dataset for training."""
    import numpy as np
    
    # Create character vocabulary
    chars = sorted(set(text))
    vocab_size = len(chars)
    log(f"Vocabulary size: {vocab_size}")
    
    # Create character to index mapping
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for i, c in enumerate(chars)}
    
    # Convert text to indices
    indices = np.array([char_to_idx[c] for c in text])
    
    # Create sequences with 50% overlap for more training data
    sequences = []
    targets = []
    
    for i in range(0, len(indices) - seq_length, seq_length // 2):
        seq = indices[i:i + seq_length]
        target = indices[i + 1:i + seq_length + 1]
        sequences.append(seq)
        targets.append(target)
    
    log(f"Number of training sequences: {len(sequences)}")
    
    return sequences, targets, vocab_size, char_to_idx, idx_to_char


def create_opt_style_model(
    vocab_size: int,
    seq_length: int = 128,
    hidden_dim: int = 768,
    num_layers: int = 4,
    num_heads: int = 8,
    intermediate_dim: int = 3072
):
    """
    Create an OPT-style transformer model for text generation.
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
    
    # Positional encoding
    positions = np.arange(seq_length)[np.newaxis, :, np.newaxis]
    position_embeddings = np.zeros((1, seq_length, hidden_dim))
    position_embeddings[:, :, :hidden_dim] = positions
    position_embeddings = keras.ops.convert_to_numpy(position_embeddings)
    x = x + position_embeddings
    
    # Transformer blocks
    for i in range(num_layers):
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=hidden_dim // num_heads,
            name=f'attention_{i}'
        )(x, x)
        
        attention_output = layers.Dropout(0.1)(attention_output)
        x = layers.Add()([x, attention_output])
        x = layers.LayerNormalization(epsilon=1e-5)(x)
        
        ffn = keras.Sequential([
            layers.Dense(intermediate_dim, activation='gelu', name=f'ffn_dense1_{i}'),
            layers.Dense(hidden_dim, name=f'ffn_dense2_{i}')
        ], name=f'ffn_{i}')
        
        ffn_output = ffn(x)
        ffn_output = layers.Dropout(0.1)(ffn_output)
        
        x = layers.Add()([x, ffn_output])
        x = layers.LayerNormalization(epsilon=1e-5)(x)
    
    # Output layer
    x = layers.Dense(hidden_dim, activation='gelu', name='output_dense')(x)
    outputs = layers.Dense(vocab_size, activation='softmax', name='logits')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='OPT_style_model')
    
    return model


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
    
    # Check if we're running with torchrun
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        
        if torch.cuda.is_available() and local_rank < gpu_count:
            torch.cuda.set_device(local_rank)
            log(f"✓ PyTorch distributed: rank={local_rank}, world_size={world_size}, device=cuda:{local_rank}")
        else:
            log(f"✓ PyTorch distributed: rank={local_rank}, world_size={world_size}, device=CPU")
    else:
        log("Running in single-process mode")
    
    log("")


def train_opt_data_parallel(
    epochs: int = 3,
    seq_length: int = 128,
    batch_size: int = 32,
    learning_rate: float = 0.0001
):
    """Train OPT-style model with DataParallel on Tiny Shakespeare dataset."""
    log_section("OPT-125M DATA PARALLEL TRAINING")
    
    import keras
    from keras import layers
    from keras.src.distribution import DataParallel, list_devices
    import numpy as np
    
    # Get devices
    devices = list_devices("gpu")
    if not devices:
        devices = ["cpu:0"]
    
    log(f"Using devices: {devices}")
    
    # Download and prepare dataset
    log("Loading Tiny Shakespeare dataset...")
    text = download_tiny_shakespeare()
    sequences, targets, vocab_size, char_to_idx, idx_to_char = prepare_text_dataset(
        text, seq_length=seq_length
    )
    
    x_train = np.array(sequences, dtype=np.int32)
    y_train = np.array(targets, dtype=np.int32)
    
    log(f"Training data shapes: x={x_train.shape}, y={y_train.shape}")
    
    # Create DataParallel distribution
    dp = DataParallel(devices=devices, auto_shard_dataset=False)
    log(f"✓ DataParallel created: mesh_shape={dp.device_mesh.shape}")
    
    # Create OPT-style model
    hidden_dim = 768
    num_layers = 4
    num_heads = 8
    
    log(f"Creating OPT-style model...")
    log(f"  - Hidden dimension: {hidden_dim}")
    log(f"  - Number of layers: {num_layers}")
    log(f"  - Number of attention heads: {num_heads}")
    
    with dp.scope():
        model = create_opt_style_model(
            vocab_size=vocab_size,
            seq_length=seq_length,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads
        )
        
        total_params = model.count_params()
        log(f"✓ Model created with {total_params:,} parameters")
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    log(f"")
    log(f"Starting DataParallel training for {epochs} epochs...")
    log(f"")
    
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        with dp.scope():
            history = model.fit(
                x_train, y_train,
                epochs=1,
                batch_size=batch_size,
                validation_split=0.1,
                verbose=1
            )
        
        epoch_time = time.time() - epoch_start
        loss = history.history['loss'][0]
        val_loss = history.history['val_loss'][0]
        
        log(f"  Epoch {epoch+1}/{epochs}: loss={loss:.4f}, val_loss={val_loss:.4f} (time={epoch_time:.1f}s)")
    
    total_time = time.time() - start_time
    
    log("")
    log(f"✓ DataParallel Training Complete!")
    log(f"  - Parameters: {total_params:,}")
    log(f"  - Final loss: {history.history['loss'][-1]:.4f}")
    log(f"  - Total time: {total_time:.1f}s")
    log("")
    
    return True


def train_opt_model_parallel(
    epochs: int = 3,
    seq_length: int = 128,
    batch_size: int = 32,
    learning_rate: float = 0.0001
):
    """Train OPT-style model with ModelParallel on Tiny Shakespeare dataset."""
    log_section("OPT-125M MODEL PARALLEL TRAINING")
    
    import torch
    import keras
    from keras import layers
    from keras.src.distribution import ModelParallel, DeviceMesh, LayoutMap, list_devices
    import numpy as np
    
    # Check GPU count
    if torch.cuda.device_count() < 2:
        log("⚠ Skipping ModelParallel: Need >= 2 GPUs")
        log(f"  Available GPUs: {torch.cuda.device_count()}")
        return False
    
    # Get devices
    devices = list_devices("gpu")
    log(f"Using devices: {devices}")
    
    # Download and prepare dataset
    log("Loading Tiny Shakespeare dataset...")
    text = download_tiny_shakespeare()
    sequences, targets, vocab_size, char_to_idx, idx_to_char = prepare_text_dataset(
        text, seq_length=seq_length
    )
    
    x_train = np.array(sequences, dtype=np.int32)
    y_train = np.array(targets, dtype=np.int32)
    
    log(f"Training data shapes: x={x_train.shape}, y={y_train.shape}")
    
    # Create 2D device mesh for model parallelism
    mesh = DeviceMesh(
        shape=(1, len(devices)),
        axis_names=["batch", "model"],
        devices=devices
    )
    log(f"✓ DeviceMesh created: shape={mesh.shape}, axes={mesh.axis_names}")
    
    # Create layout map for sharding
    layout_map = LayoutMap(mesh)
    
    # Shard weights on model axis
    layout_map[".*token_embedding"] = (None, "model")
    layout_map[".*attention.*kernel"] = (None, "model")
    layout_map[".*ffn.*kernel"] = (None, "model")
    layout_map[".*output_dense.*kernel"] = (None, "model")
    layout_map[".*bias"] = ("model",)
    
    log("✓ LayoutMap configured for weight sharding:")
    for key in list(layout_map.keys())[:3]:
        layout = layout_map[key]
        log(f"  - {key}: axes={layout.axes}")
    log("  - ... (more patterns)")
    
    # Create ModelParallel distribution
    mp = ModelParallel(
        layout_map=layout_map,
        batch_dim_name="batch",
        auto_shard_dataset=False
    )
    log(f"✓ ModelParallel created: batch_dim={mp.batch_dim_name}")
    
    # Create larger model to benefit from sharding
    hidden_dim = 1024
    num_layers = 6
    num_heads = 8
    
    log(f"Creating OPT-style model for sharding...")
    log(f"  - Hidden dimension: {hidden_dim}")
    log(f"  - Number of layers: {num_layers}")
    log(f"  - Sharding: weights split across {len(devices)} GPUs")
    
    with mp.scope():
        model = create_opt_style_model(
            vocab_size=vocab_size,
            seq_length=seq_length,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads
        )
        
        total_params = model.count_params()
        log(f"✓ Model created with {total_params:,} parameters")
        log(f"  (Weight sharding across {len(devices)} GPUs)")
        
        # Show sharding info
        for i, layer in enumerate(model.layers):
            if hasattr(layer, 'kernel') and layer.kernel is not None:
                log(f"  Layer {i}: {layer.name}, kernel_shape={layer.kernel.shape}")
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    log(f"")
    log(f"Starting ModelParallel training for {epochs} epochs...")
    log(f"  - Sharding: weights split across {len(devices)} GPUs")
    log(f"")
    
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        with mp.scope():
            history = model.fit(
                x_train, y_train,
                epochs=1,
                batch_size=batch_size,
                validation_split=0.1,
                verbose=1
            )
        
        epoch_time = time.time() - epoch_start
        loss = history.history['loss'][0]
        val_loss = history.history['val_loss'][0]
        
        log(f"  Epoch {epoch+1}/{epochs}: loss={loss:.4f}, val_loss={val_loss:.4f} (time={epoch_time:.1f}s)")
    
    total_time = time.time() - start_time
    
    log("")
    log(f"✓ ModelParallel Training Complete!")
    log(f"  - Parameters: {total_params:,}")
    log(f"  - Device mesh: {mesh.shape}")
    log(f"  - Final loss: {history.history['loss'][-1]:.4f}")
    log(f"  - Total time: {total_time:.1f}s")
    log(f"  - Sharding: weights split across {len(devices)} GPUs")
    log("")
    
    return True


def main():
    """Main entry point."""
    import torch
    import torch.distributed as dist
    from keras.src.distribution import initialize
    
    # Setup environment
    setup_environment()
    
    # Initialize Keras distribution
    initialize()
    
    # Run DataParallel test (always runs)
    train_opt_data_parallel(
        epochs=3,
        seq_length=128,
        batch_size=32,
        learning_rate=0.0001
    )
    
    # Run ModelParallel test (only if >= 2 GPUs)
    if torch.cuda.device_count() >= 2:
        train_opt_model_parallel(
            epochs=3,
            seq_length=128,
            batch_size=32,
            learning_rate=0.0001
        )
    else:
        log_section("MODEL PARALLEL TEST (SKIPPED)")
        log("Need >= 2 GPUs for ModelParallel test")
        log("Run on a machine with 2+ GPUs to test ModelParallel")
        log("")
    
    # Cleanup
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    
    log_section("ALL TESTS COMPLETED")
    log("✓ OPT-125M distributed training verification complete!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

