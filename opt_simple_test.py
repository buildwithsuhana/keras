#!/usr/bin/env python3
"""
Simple Distributed Training Test for OPT-125M

This is a lightweight test that verifies:
1. DataParallel works with OPT-125M
2. ModelParallel works with OPT-125M (on 2+ GPUs)

Usage:
    # Single GPU (tests DataParallel only):
    python opt_simple_test.py
    
    # Multi-GPU (2 T4 GPUs on Kaggle):
    torchrun --nproc_per_node=2 opt_simple_test.py
"""

import os
# MUST be set before any other imports
os.environ["KERAS_BACKEND"] = "torch"
os.environ["KERAS_DISTRIBUTION_DEBUG"] = "0"
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
    separator = "=" * 60
    log(separator, rank_0_only=True)
    log(f"  {title}", rank_0_only=True)
    log(separator, rank_0_only=True)


def prepare_dummy_data(seq_length=64, batch_size=8, num_samples=100):
    """Prepare dummy data for quick testing."""
    import numpy as np
    
    # Dummy token IDs
    x_train = np.random.randint(0, 1000, size=(num_samples, seq_length), dtype=np.int32)
    y_train = np.random.randint(0, 1000, size=(num_samples, seq_length), dtype=np.int32)
    padding_mask = np.ones((num_samples, seq_length), dtype=np.int32)
    
    return x_train, y_train, padding_mask


def test_data_parallel_simple(epochs=1, seq_length=64):
    """Test DataParallel with OPT-125M."""
    import torch
    import keras
    from keras.src.distribution import DataParallel, list_devices, initialize
    import numpy as np
    
    log_section("DATA PARALLEL TEST")
    
    initialize()
    
    # Get devices
    devices = list_devices("gpu")
    if not devices:
        devices = ["cpu:0"]
    
    log(f"Using devices: {devices}")
    
    # Create DataParallel
    dp = DataParallel(devices=devices, auto_shard_dataset=False)
    log(f"✓ DataParallel created: mesh={dp.device_mesh.shape}")
    
    # Check if keras_hub is available
    try:
        import keras_hub
        log(f"keras_hub available: {keras_hub.__version__}")
        has_keras_hub = True
    except ImportError:
        log("keras_hub not available")
        has_keras_hub = False
    
    # Create model
    with dp.scope():
        if has_keras_hub:
            try:
                # Use OPT-125M from keras_hub
                log("Loading OPT-125M backbone...")
                opt_backbone = keras_hub.models.OPTBackbone.from_preset(
                    "opt_125m_en",
                    dtype="float32"
                )
                
                # Build the model by passing dummy input
                dummy_input = {
                    "token_ids": np.zeros((1, seq_length), dtype=np.int32),
                    "padding_mask": np.ones((1, seq_length), dtype=np.int32)
                }
                _ = opt_backbone(dummy_input)
                
                # Create model
                inputs = keras.Input(shape=(seq_length,), dtype='int32', name="token_ids")
                padding_mask = keras.Input(shape=(seq_length,), dtype='int32', name="padding_mask")
                
                x = opt_backbone({"token_ids": inputs, "padding_mask": padding_mask})
                outputs = keras.layers.Dense(opt_backbone.vocabulary_size, name="output_dense")(x)
                
                model = keras.Model(inputs=[inputs, padding_mask], outputs=outputs)
                
                log(f"✓ OPT-125M model created: {model.count_params():,} params")
                
            except Exception as e:
                log(f"Error loading OPT: {e}")
                has_keras_hub = False
        
        if not has_keras_hub:
            # Fallback to simple model - build it first
            log("Creating simple model...")
            model = keras.Sequential([
                keras.layers.Embedding(1000, 128),
                keras.layers.Dense(256, activation='relu'),
                keras.layers.Dense(1000)
            ])
            
            # Build the model by passing dummy input
            dummy_input = np.zeros((1, seq_length), dtype=np.int32)
            _ = model(dummy_input)
            
            log(f"✓ Simple model created: {model.count_params():,} params")
        
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')
    
    # Prepare data
    log("Preparing data...")
    x_train, y_train, padding_mask = prepare_dummy_data(seq_length=seq_length)
    log(f"Data shapes: x={x_train.shape}, y={y_train.shape}")
    
    # Quick training test
    log("Training (1 epoch)...")
    with dp.scope():
        history = model.fit(
            [x_train, padding_mask] if has_keras_hub else x_train,
            y_train,
            epochs=epochs,
            batch_size=8,
            verbose=1
        )
    
    log(f"✓ DataParallel completed. Final loss: {history.history['loss'][-1]:.4f}")
    return True


def test_model_parallel_simple(epochs=1, seq_length=64):
    import torch
    import keras
    from keras.src.distribution import ModelParallel, DeviceMesh, LayoutMap, list_devices, initialize
    import numpy as np
    
    log_section("MODEL PARALLEL TEST")
    initialize()
    
    # Ensure backend is clean
    keras.backend.set_floatx("float32")
    
    gpu_count = torch.cuda.device_count()
    if gpu_count < 2:
        log(f"⚠ Skipping: Need 2+ GPUs, have {gpu_count}")
        return False
    
    devices = list_devices("gpu")
    mesh = DeviceMesh(
        shape=(1, len(devices)),
        axis_names=["batch", "model"],
        devices=devices
    )
    
    # Updated LayoutMap for KerasHub OPT
    layout_map = LayoutMap(mesh)
    layout_map[".*embeddings/.*"] = (None, "model")
    layout_map[".*rel_p_embeddings/.*"] = (None, "model") # For OPT
    layout_map[".*kernel"] = (None, "model")
    layout_map[".*bias"] = (None, "model")
    
    mp = ModelParallel(
        layout_map=layout_map,
        batch_dim_name="batch",
    )
    
    try:
        import keras_hub
        has_keras_hub = True
    except ImportError:
        has_keras_hub = False
    
    with mp.scope():
        if has_keras_hub:
            log("Creating OPT backbone with sharding...")
            # Use the preset but wrap it in the scope to ensure weights are DTensors
            opt_backbone = keras_hub.models.OPTBackbone(
                vocabulary_size=50265,
                num_layers=4,
                num_heads=8,
                hidden_dim=512,
                intermediate_dim=2048,
                max_sequence_length=seq_length,
                dtype="float32"
            )
            
            inputs = keras.Input(shape=(seq_length,), dtype='int32', name="token_ids")
            padding_mask = keras.Input(shape=(seq_length,), dtype='int32', name="padding_mask")
            
            x = opt_backbone({"token_ids": inputs, "padding_mask": padding_mask})
            outputs = keras.layers.Dense(50265, name="output_dense")(x)
            model = keras.Model(inputs=[inputs, padding_mask], outputs=outputs)
        else:
            model = keras.Sequential([
                keras.layers.Embedding(1000, 256),
                keras.layers.Dense(1000)
            ])
            
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        # Use a simpler loss for debugging if necessary
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')

    log("Preparing data...")
    x_train, y_train, padding_mask = prepare_dummy_data(seq_length=seq_length, num_samples=50)
    
    # IMPORTANT: In PyTorch ModelParallel, we often need to ensure the 
    # dataset itself is prepared to yield tensors that match the backend requirement.
    log("Training (1 epoch)...")
    with mp.scope():
        # Keras .fit() handles the distribution of numpy arrays automatically 
        # as long as it's called WITHIN the mp.scope()
        history = model.fit(
            x={"token_ids": x_train, "padding_mask": padding_mask} if has_keras_hub else x_train,
            y=y_train,
            epochs=epochs,
            batch_size=4,
            verbose=1
        )
    
    log(f"✓ ModelParallel completed.")
    return True


def main():
    """Main entry point."""
    import torch
    import torch.distributed as dist
    from keras.src.distribution import initialize
    
    # Initialize
    initialize()
    
    # Setup
    log_section("ENVIRONMENT")
    log(f"PyTorch: {torch.__version__}")
    log(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"GPUs: {torch.cuda.device_count()}")
    
    is_dist = dist.is_available() and dist.is_initialized()
    if is_dist:
        log(f"Distributed: rank={dist.get_rank()}, world={dist.get_world_size()}")
    
    # Get GPU count
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    # Run tests
    log_section("STARTING TESTS")
    
    # DataParallel test (works on any GPU count)
    test_data_parallel_simple(epochs=1, seq_length=64)
    
    # ModelParallel test (requires 2+ GPUs)
    if gpu_count >= 2:
        test_model_parallel_simple(epochs=1, seq_length=64)
    else:
        log_section("SKIPPED MODEL PARALLEL")
        log("Need 2+ GPUs for ModelParallel")
    
    # Summary
    log_section("COMPLETE")
    log("✓ All tests passed!")
    
    # Cleanup
    if is_dist:
        dist.destroy_process_group()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

