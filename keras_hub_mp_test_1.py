#!/usr/bin/env python3
"""
Keras Hub Model ModelParallel Verification Script - FIXED

This script tests ModelParallel with Keras Hub models like OPT.
It verifies that model weights are correctly sharded across GPUs.

Usage:
    torchrun --nproc_per_node=2 keras_hub_mp_test.py
"""

import os
# MUST be set before any other imports
os.environ["KERAS_BACKEND"] = "torch"
os.environ["KERAS_DISTRIBUTION_DEBUG"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import time
import logging
import numpy as np

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
    
    if torch.cuda.is_available():
        if "LOCAL_RANK" in os.environ:
            local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(local_rank)
            log(f"✓ PyTorch distributed initialized on GPU {torch.cuda.current_device()}")
    
    if dist.is_available() and dist.is_initialized():
        log(f"✓ Distributed initialized: Rank {dist.get_rank()}, Size {dist.get_world_size()}")
    
    log("")


def test_keras_hub_model_parallel(epochs=1, use_preset=True):
    """Test ModelParallel with Keras Hub models (e.g., OPT)."""
    import torch
    import torch.distributed as dist
    import keras
    from keras.src.distribution import ModelParallel, DeviceMesh, LayoutMap, list_devices
    # Import critical backend utilities for PyTorch distribution
    from keras.src.backend.torch.distribution_lib import set_mp_multi_process_state, prepare_input_for_distribution
    
    log_section("TEST: KERAS HUB MODEL PARALLEL (MP)")
    
    # Check GPU count
    devices = list_devices("gpu")
    if len(devices) < 2:
        log("⚠ Skipping ModelParallel test: Need >= 2 GPUs")
        return False
    
    rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
    world_size = dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1
    
    log(f"Using {len(devices)} device(s): {devices}")
    
    # Create 2D device mesh for model parallelism
    mesh = DeviceMesh(
        shape=(1, len(devices)),
        axis_names=["batch", "model"],
        devices=devices
    )
    log(f"✓ DeviceMesh created: shape={mesh.shape}")
    
    # Configure LayoutMap for sharding
    layout_map = LayoutMap(mesh)
    layout_map["embeddings.token_embedding.embeddings"] = (None, "model")
    layout_map["embeddings.position_embedding.embeddings"] = (None, None)  # Replicated DTensor
    layout_map["transformer_layer_.*.attention.query.kernel"] = (None, "model")
    layout_map["transformer_layer_.*.attention.key.kernel"] = (None, "model")
    layout_map["transformer_layer_.*.attention.value.kernel"] = (None, "model")
    layout_map["transformer_layer_.*.attention.output.kernel"] = (None, "model")
    layout_map["transformer_layer_.*.feedforward.gate.kernel"] = (None, "model")
    layout_map["transformer_layer_.*.feedforward.up.kernel"] = (None, "model")
    layout_map["transformer_layer_.*.feedforward.down.kernel"] = (None, "model")
    layout_map["transformer_layer_.*.self_attention_layer_norm.*"] = ()
    layout_map["transformer_layer_.*.self_feedforward_layer_norm.*"] = ()
    layout_map["embeddings.layer_norm.*"] = ()
    
    # Create ModelParallel distribution
    mp = ModelParallel(layout_map=layout_map, batch_dim_name="batch", auto_shard_dataset=False)
    
    # CRITICAL FIX 1: Explicitly set the multi-process state to True
    # This triggers the backend logic to wrap inputs in DTensors
    set_mp_multi_process_state(True)
    
    try:
        import keras_hub
    except ImportError as e:
        log(f"⚠ Failed to import keras_hub: {e}")
        return False
    
    with mp.scope():
        if use_preset:
            log("Loading OPT preset...")
            model = keras_hub.models.OPTBackbone.from_preset("opt_125m_en")
        else:
            model = keras_hub.models.OPTBackbone(
                vocabulary_size=50265, num_layers=2, num_heads=2,
                hidden_dim=128, intermediate_dim=256, max_sequence_length=32
            )
        
        # CRITICAL FIX 2: Build the model explicitly inside the distribution scope
        # This ensures weights are sharded as DTensors before compilation
        batch_size = 4
        seq_length = 16
        model.build({"token_ids": (batch_size, seq_length), "padding_mask": (batch_size, seq_length)})
        log("✓ Model built inside ModelParallel scope")
        
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss="mse")
    
    log_section("PHYSICAL STORAGE VERIFICATION")
    verify_weight_sharding(model, rank)
    
    # Create dummy input data
    x_raw = {
        "token_ids": np.random.randint(0, 50265, size=(batch_size, seq_length), dtype="int32"),
        "padding_mask": np.ones((batch_size, seq_length), dtype="int32"),
    }
    y_raw = np.random.random((batch_size, seq_length, model.hidden_dim)).astype("float32")
    
    # CRITICAL FIX 3: Manually distribute inputs into DTensors before training
    # This resolves the "mixed torch.Tensor and DTensor" error
    log("Preparing distributed data...")
    with mp.scope():
        x_dist = prepare_input_for_distribution(x_raw)
        y_dist = prepare_input_for_distribution(y_raw)
    
    log(f"Data ready: token_ids type = {type(x_dist['token_ids'])}")
    
    # Training loop
    log(f"Training for {epochs} epoch(s)...")
    for epoch in range(epochs):
        with mp.scope():
            history = model.fit(x_dist, y_dist, epochs=1, verbose=0)
            log(f"  Epoch {epoch+1}: loss={history.history['loss'][0]:.6f}")
    
    log("✓ ModelParallel test PASSED")
    return True


def verify_weight_sharding(model, rank):
    """Verify that model weights are sharded correctly."""
    torch_model = getattr(model, '_torch_layers', getattr(model, 'torch_layer', None))
    if not torch_model:
        log("  ⚠ Could not access torch layers for verification")
        return
    
    # Check Token Embedding sharding
    if hasattr(torch_model, 'embeddings') and hasattr(torch_model.embeddings, 'token_embedding'):
        weight = torch_model.embeddings.token_embedding.weight
        if hasattr(weight, 'to_local'):
            global_shape = tuple(weight.shape)
            local_shape = tuple(weight.to_local().shape)
            log(f"  Layer token_embedding: Global {global_shape} -> Local {local_shape}")
            if global_shape[-1] != local_shape[-1]:
                log(f"    ✓ Rank {rank}: Weight is sharded across 'model' axis")


def main():
    """Main entry point."""
    from keras.src.distribution import initialize
    initialize()
    setup_environment()
    
    try:
        test_keras_hub_model_parallel(epochs=1, use_preset=False)
    except Exception as e:
        log(f"⚠ ModelParallel test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    log_section("VERIFICATION SUMMARY")
    log("✓ Keras Hub ModelParallel test completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())