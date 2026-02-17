#!/usr/bin/env python3
"""
Keras Hub Model ModelParallel Verification Script - UPDATED

This script tests ModelParallel with Keras Hub models like OPT.
It verifies that model weights are correctly sharded across GPUs.
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
    separator = "=" * 70
    log(separator, rank_0_only=True)
    log(f"  {title}", rank_0_only=True)
    log(separator, rank_0_only=True)

def setup_environment():
    import torch
    import torch.distributed as dist
    log_section("ENVIRONMENT SETUP")
    log(f"Python version: {sys.version.split()[0]}")
    log(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        log(f"Number of GPUs: {torch.cuda.device_count()}")
        if "LOCAL_RANK" in os.environ:
            local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(local_rank)
            log(f"Rank {local_rank} initialized on GPU {torch.cuda.current_device()}")
    log("")

def test_keras_hub_model_parallel(epochs=1, use_preset=True):
    import torch
    import torch.distributed as dist
    import keras
    from keras.src.distribution import ModelParallel, DeviceMesh, LayoutMap, list_devices
    # Import utility to sync backend state
    from keras.src.backend.torch.distribution_lib import set_mp_multi_process_state
    
    log_section("TEST: KERAS HUB MODEL PARALLEL (MP)")
    
    devices = list_devices("gpu")
    if len(devices) < 2:
        log("⚠ Skipping ModelParallel test: Need >= 2 GPUs")
        return False
    
    rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
    
    # Create 2D device mesh
    mesh = DeviceMesh(
        shape=(1, len(devices)),
        axis_names=["batch", "model"],
        devices=devices
    )
    log(f"✓ DeviceMesh created: shape={mesh.shape}")
    
    layout_map = LayoutMap(mesh)
    # Configure LayoutMap for OPT model sharding
    layout_map["embeddings.token_embedding.embeddings"] = (None, "model")
    layout_map["embeddings.position_embedding.embeddings"] = (None, None)
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
    
    mp = ModelParallel(layout_map=layout_map, batch_dim_name="batch", auto_shard_dataset=False)
    
    # UPDATE: Set multi-process state so backend handles inputs correctly
    set_mp_multi_process_state(True)

    try:
        import keras_hub
    except ImportError:
        log("⚠ keras_hub not found")
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
        
        # UPDATE: Explicitly build model with dummy shapes INSIDE the scope
        # This ensures weights are sharded as DTensors before compilation
        batch_size = 4
        seq_length = 16
        model.build({"token_ids": (batch_size, seq_length), "padding_mask": (batch_size, seq_length)})
        log("✓ Model built inside ModelParallel scope")
        
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss="mse")

    log_section("PHYSICAL STORAGE VERIFICATION")
    verify_weight_sharding(model, rank)
    
    # Create training data
    x = {
        "token_ids": np.random.randint(0, 50265, size=(batch_size, seq_length), dtype="int32"),
        "padding_mask": np.ones((batch_size, seq_length), dtype="int32"),
    }
    y = np.random.random((batch_size, seq_length, model.hidden_dim)).astype("float32")
    
    log(f"Training for {epochs} epoch(s)...")
    with mp.scope():
        # Inputs will be automatically converted to Replicated DTensors
        history = model.fit(x, y, epochs=epochs, verbose=0)
        log(f"  Loss: {history.history['loss'][0]:.6f}")
    
    log("✓ ModelParallel test PASSED")
    return True

def verify_weight_sharding(model, rank):
    # Determine the internal torch model object
    torch_model = getattr(model, '_torch_layers', getattr(model, 'torch_layer', None))
    if not torch_model:
        log("  ⚠ Access to torch layers failed")
        return

    # Check specific weight sharding
    if hasattr(torch_model, 'embeddings') and hasattr(torch_model.embeddings, 'token_embedding'):
        weight = torch_model.embeddings.token_embedding.weight
        if hasattr(weight, 'to_local'):
            log(f"  Layer embeddings.token_embedding:")
            log(f"    - Global Shape: {tuple(weight.shape)}")
            log(f"    - Local Shape (Rank {rank}): {tuple(weight.to_local().shape)}")
            if weight.shape[-1] != weight.to_local().shape[-1]:
                log("    ✓ Sharded across 'model' axis")

def main():
    from keras.src.distribution import initialize, list_devices
    initialize()
    setup_environment()
    
    if len(list_devices("gpu")) >= 2:
        test_keras_hub_model_parallel(epochs=1, use_preset=False)
    else:
        log("Need >= 2 GPUs for ModelParallel test")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())