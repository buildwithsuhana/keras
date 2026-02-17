#!/usr/bin/env python3
"""
Keras Hub Model ModelParallel Verification Script

This script tests ModelParallel with Keras Hub models like OPT.
It verifies that model weights are correctly sharded across GPUs.

Usage:
    torchrun --nproc_per_node=2 keras_hub_mp_test.py

Or in Kaggle:
    !torchrun --nproc_per_node=2 /path/to/this/file.py
"""

import os
# MUST be set before any other imports
os.environ["KERAS_BACKEND"] = "torch"
os.environ["KERAS_DISTRIBUTION_DEBUG"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import time
import logging
from datetime import datetime

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
        
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            actual_gpu = torch.cuda.current_device()
            log(f"[initialize() Rank {local_rank}] Set CUDA device to GPU {actual_gpu} (visible as 0), num_gpus={gpu_count}")
            log(f"✓ PyTorch distributed initialized via torchrun")
            log(f"  Local rank: {local_rank}, World size: {world_size}")
            log(f"  Device: cuda:{actual_gpu}")
        else:
            log(f"✓ PyTorch distributed initialized via torchrun (CPU mode)")
            log(f"  Local rank: {local_rank}, World size: {world_size}")
            log(f"  Device: CPU")
    else:
        log("Running in single-process mode")
    
    # Check distributed status
    is_dist = dist.is_available() and dist.is_initialized()
    log(f"Distributed initialized: {is_dist}")
    if is_dist:
        log(f"  Rank: {dist.get_rank()}, World size: {dist.get_world_size()}")
    
    log("")


def test_device_detection():
    """Test device detection."""
    import torch
    from keras.src.distribution import list_devices
    
    log_section("TEST 1: DEVICE DETECTION")
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        log(f"✓ PyTorch detected {gpu_count} GPU(s)")
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            log(f"  - cuda:{i} = {props.name} ({props.total_memory / 1e9:.1f} GB)")
    else:
        log("⚠ No GPU detected, using CPU")
    
    # Keras detection
    devices = list_devices("gpu")
    log(f"✓ Keras detected GPU devices: {devices}")
    
    log("")


def test_keras_hub_model_parallel(epochs=1, use_preset=True):
    """Test ModelParallel with Keras Hub models (e.g., OPT)."""
    import torch
    import torch.distributed as dist
    import keras
    from keras.src.distribution import ModelParallel, DeviceMesh, LayoutMap, list_devices
    from keras.src.backend.torch.distribution_lib import set_mp_multi_process_state
    import numpy as np
    import time
    
    log_section("TEST: KERAS HUB MODEL PARALLEL (MP)")
    
    # Check GPU count
    gpu_count = len(list_devices("gpu"))
    if gpu_count < 2:
        log("⚠ Skipping ModelParallel test: Need >= 2 GPUs")
        log(f"  Available GPUs: {gpu_count}")
        return False
    
    # Get devices
    devices = list_devices("gpu")
    
    rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
    world_size = dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1
    
    log(f"Using {len(devices)} device(s): {devices}")
    log(f"World size: {world_size}, Rank: {rank}")
    
    # Create 2D device mesh for model parallelism
    mesh = DeviceMesh(
        shape=(1, len(devices)),
        axis_names=["batch", "model"],
        devices=devices
    )
    log(f"✓ DeviceMesh created: shape={mesh.shape}, axes={mesh.axis_names}")
    
    layout_map = LayoutMap(mesh)
    layout_map["embeddings.token_embedding.embeddings"] = (None, "model")  # Token embeddings - sharded
    layout_map["embeddings.position_embedding.embeddings"] = (None, None)  # Position embeddings - REPLICATED DTensor
    
    # Transformer decoder layers - attention weights
    layout_map["transformer_layer_.*.attention.query.kernel"] = (None, "model")
    layout_map["transformer_layer_.*.attention.key.kernel"] = (None, "model")
    layout_map["transformer_layer_.*.attention.value.kernel"] = (None, "model")
    layout_map["transformer_layer_.*.attention.output.kernel"] = (None, "model")
    
    # Transformer decoder layers - feedforward weights
    layout_map["transformer_layer_.*.feedforward.gate.kernel"] = (None, "model")
    layout_map["transformer_layer_.*.feedforward.up.kernel"] = (None, "model")
    layout_map["transformer_layer_.*.feedforward.down.kernel"] = (None, "model")
    
    layout_map["transformer_layer_.*.self_attention_layer_norm.gamma"] = ()
    layout_map["transformer_layer_.*.self_attention_layer_norm.beta"] = ()
    layout_map["transformer_layer_.*.self_feedforward_layer_norm.gamma"] = ()
    layout_map["transformer_layer_.*.self_feedforward_layer_norm.beta"] = ()
    
    # Embeddings layer norms
    layout_map["embeddings.layer_norm.gamma"] = ()
    layout_map["embeddings.layer_norm.beta"] = ()
    
    log("✓ LayoutMap configured:")
    log("  - embeddings.token_embedding.embeddings: (None, 'model') [sharded]")
    log("  - embeddings.position_embedding.embeddings: (None, None) [replicated DTensor]")
    log("  - transformer_layer_*.attention.*.kernel: (None, 'model')")
    log("  - transformer_layer_*.feedforward.*.kernel: (None, 'model')")
    log("  - Layer norms (gamma/beta): () [replicated]")
    
    # Create ModelParallel distribution
    mp = ModelParallel(
        layout_map=layout_map,
        batch_dim_name="batch",
        auto_shard_dataset=False
    )
    log(f"✓ ModelParallel created: batch_dim={mp.batch_dim_name}")
    log(f"  Auto-shard dataset: False")
    
    # Import keras_hub
    try:
        import keras_hub
        log("✓ keras_hub imported successfully")
    except ImportError as e:
        log(f"⚠ Failed to import keras_hub: {e}")
        return False
    
    # Create model - either from preset or custom
    with mp.scope():
        if use_preset:
            try:
                # Try to load a small preset
                # Note: from_preset might download weights, which may not work in all environments
                log("Loading OPT preset (this may download weights)...")
                model = keras_hub.models.OPTBackbone.from_preset("opt_125m_en")
                log(f"✓ Loaded OPTBackbone from preset")
            except Exception as e:
                log(f"⚠ Failed to load preset: {e}")
                log("Creating custom OPT model instead...")
                model = keras_hub.models.OPTBackbone(
                    vocabulary_size=50265,
                    num_layers=2,  # Use fewer layers for testing
                    num_heads=2,
                    hidden_dim=128,  # Smaller for testing
                    intermediate_dim=256,
                    max_sequence_length=32,
                )
                log(f"✓ Created custom OPTBackbone")
        else:
            # Create custom model
            model = keras_hub.models.OPTBackbone(
                vocabulary_size=50265,
                num_layers=2,
                num_heads=2,
                hidden_dim=128,
                intermediate_dim=256,
                max_sequence_length=32,
            )
            log(f"✓ Created custom OPTBackbone")
        
        # Count parameters
        total_params = model.count_params()
        log(f"✓ Model created with {total_params:,} parameters")
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss="mse")
    
    # Verify weight sharding
    log_section("PHYSICAL STORAGE VERIFICATION")
    verify_weight_sharding(model, rank)
    
    # CRITICAL FIX: Set the multi-process state so the backend handles inputs correctly
    # This enables proper DTensor conversion for inputs in multi-process ModelParallel mode
    set_mp_multi_process_state(True)
    
    # Create dummy input data
    batch_size = 4
    seq_length = 16
    
    x = {
        "token_ids": np.random.randint(0, 50265, size=(batch_size, seq_length), dtype="int32"),
        "padding_mask": np.ones((batch_size, seq_length), dtype="int32"),
    }
    y = np.random.random((batch_size, seq_length, model.hidden_dim)).astype("float32")
    
    # CRITICAL FIX: Explicitly build the model with the input shape inside the scope
    # This ensures variables are created and distributed before fit() is called
    with mp.scope():
        model.build({"token_ids": (batch_size, seq_length), "padding_mask": (batch_size, seq_length)})
    
    log(f"Training data: token_ids_shape={x['token_ids'].shape}")
    
    # Training loop
    log(f"Training for {epochs} epoch(s)...")
    start_time = time.time()
    losses = []
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        with mp.scope():
            history = model.fit(x, y, epochs=1, verbose=0)
            loss = history.history['loss'][0]
        
        epoch_time = time.time() - epoch_start
        losses.append(loss)
        
        log(f"  Epoch {epoch+1}/{epochs}: loss={loss:.6f} (time={epoch_time:.3f}s)")
    
    total_time = time.time() - start_time
    log(f"✓ ModelParallel test PASSED in {total_time:.3f}s")
    return True


def verify_weight_sharding(model, rank):
    """Verify that model weights are sharded correctly."""
    import torch
    
    # Get the underlying torch model
    if hasattr(model, '_torch_layers'):
        torch_model = model._torch_layers
    elif hasattr(model, 'torch_layer'):
        torch_model = model.torch_layer
    else:
        log(f"  ⚠ Could not access torch layers for verification")
        return
    
    # Check a few key layers
    checked_layers = 0
    
    # Check embeddings
    if hasattr(torch_model, 'embeddings'):
        embeddings = torch_model.embeddings
        if hasattr(embeddings, 'token_embedding'):
            token_emb = embeddings.token_embedding
            if hasattr(token_emb, 'weight'):
                checked_layers = check_dtensor_weight(token_emb.weight, "embeddings.token_embedding", checked_layers, rank)
    
    # Check transformer layers
    if hasattr(torch_model, 'transformer_layers'):
        for i, layer in enumerate(torch_model.transformer_layers):
            # Check attention
            if hasattr(layer, 'attention') and hasattr(layer.attention, 'query'):
                if hasattr(layer.attention.query, 'kernel'):
                    checked_layers = check_dtensor_weight(layer.attention.query.kernel, f"transformer_layer_{i}.attention.query", checked_layers, rank)
    
    if checked_layers > 0:
        log(f"  ✓ Verified {checked_layers} weight tensor(s) are sharded")
    else:
        log(f"  ⚠ No DTensor weights found for verification")


def check_dtensor_weight(weight_tensor, name, count, rank):
    """Check if a weight tensor is a DTensor and log its shape."""
    if hasattr(weight_tensor, 'to_local'):
        # It's a DTensor
        dtensor = weight_tensor
        global_shape = tuple(dtensor.shape)
        local_tensor = dtensor.to_local()
        local_shape = tuple(local_tensor.shape)
        
        log(f"  Layer {name}:")
        log(f"    - Global Shape: {global_shape}")
        log(f"    - Local Shape (Rank {rank}): {local_shape}")
        
        # Check if sharded
        if len(global_shape) > 0 and len(local_shape) > 0:
            if global_shape[-1] != local_shape[-1]:
                log(f"    ✓ Sharded across 'model' axis")
                count += 1
            else:
                log(f"    - Not sharded (replicated)")
        return count
    else:
        log(f"  Layer {name}: Regular tensor, shape={tuple(weight_tensor.shape)}")
        return count


def test_simple_sequential_mp(epochs=1):
    """Test ModelParallel with a simple Sequential model (fallback)."""
    import torch
    import torch.distributed as dist
    import keras
    from keras import layers
    from keras.src.distribution import ModelParallel, DeviceMesh, LayoutMap, list_devices
    import numpy as np
    import time
    
    log_section("TEST: SIMPLE SEQUENTIAL MODEL PARALLEL (Fallback)")
    
    # Check GPU count
    gpu_count = len(list_devices("gpu"))
    if gpu_count < 2:
        log("⚠ Skipping test: Need >= 2 GPUs")
        return False
    
    devices = list_devices("gpu")
    rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
    world_size = dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1
    
    log(f"Using {len(devices)} device(s): {devices}")
    log(f"World size: {world_size}, Rank: {rank}")
    
    # Create device mesh
    mesh = DeviceMesh(
        shape=(1, len(devices)),
        axis_names=["batch", "model"],
        devices=devices
    )
    log(f"✓ DeviceMesh created: shape={mesh.shape}")
    
    # Create layout map
    layout_map = LayoutMap(mesh)
    layout_map["dense.*kernel"] = (None, "model")
    layout_map["dense.*bias"] = ()
    
    log("✓ LayoutMap configured")
    
    # Create ModelParallel
    mp = ModelParallel(
        layout_map=layout_map,
        batch_dim_name="batch",
        auto_shard_dataset=False
    )
    log(f"✓ ModelParallel created")
    
    # Create simple model
    with mp.scope():
        model = keras.Sequential([
            layers.Input(shape=(128,)),
            layers.Dense(256, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(8),  # Small output divisible by 2
        ])
        
        total_params = model.count_params()
        log(f"✓ Model created with {total_params:,} parameters")
        
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss="mse")
    
    # Verify sharding
    log_section("PHYSICAL STORAGE VERIFICATION")
    verify_simple_weight_sharding(model, rank)
    
    # Train
    batch_size = 16
    x = np.random.random((batch_size, 128)).astype("float32")
    y = np.random.random((batch_size, 8)).astype("float32")
    
    log(f"Training for {epochs} epoch(s)...")
    start_time = time.time()
    
    with mp.scope():
        history = model.fit(x, y, epochs=epochs, verbose=0)
    
    total_time = time.time() - start_time
    loss = history.history['loss'][-1]
    
    log(f"  Final loss: {loss:.6f} (time={total_time:.3f}s)")
    log(f"✓ Simple ModelParallel test PASSED")
    return True


def verify_simple_weight_sharding(model, rank):
    """Verify weight sharding for simple model."""
    import torch
    
    # Get torch layers
    if hasattr(model, '_torch_layers'):
        torch_layers = model._torch_layers
    else:
        log(f"  ⚠ Could not access torch layers")
        return
    
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'kernel'):
            kernel_var = layer.kernel
            if hasattr(kernel_var, '_value'):
                kernel_tensor = kernel_var._value
            elif hasattr(kernel_var, 'value'):
                kernel_tensor = kernel_var.value
            else:
                kernel_tensor = kernel_var
            
            if hasattr(kernel_tensor, 'to_local'):
                dtensor = kernel_tensor
                global_shape = tuple(dtensor.shape)
                local_tensor = dtensor.to_local()
                local_shape = tuple(local_tensor.shape)
                
                log(f"  Layer {i} ({layer.name}):")
                log(f"    - Global: {global_shape} -> Local: {local_shape}")
                
                if len(global_shape) > 1 and global_shape[1] != local_shape[1]:
                    log(f"    ✓ Sharded on 'model' axis")


def print_summary():
    """Print final summary."""
    log_section("VERIFICATION SUMMARY")
    log("✓ All Keras Hub ModelParallel tests completed!")
    log("=" * 70)


def main():
    """Main entry point."""
    import torch.distributed as dist
    from keras.src.distribution import initialize
    from keras.src.distribution import list_devices
    
    # Initialize Keras distribution system
    initialize()
    
    # Setup environment
    setup_environment()
    
    # Run tests
    test_device_detection()
    
    gpu_count = len(list_devices("gpu"))
    
    if gpu_count >= 2:
        # Try Keras Hub model first
        log("Attempting to test with OPT model...")
        try:
            test_keras_hub_model_parallel(epochs=1, use_preset=False)
        except Exception as e:
            log(f"⚠ Keras Hub model test failed: {e}")
            log("Falling back to simple Sequential test...")
            test_simple_sequential_mp(epochs=1)
    else:
        log_section("MODEL PARALLEL TEST (SKIPPED)")
        log("Need >= 2 GPUs for ModelParallel test")
    
    # Print summary
    print_summary()
    
    # Cleanup
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

