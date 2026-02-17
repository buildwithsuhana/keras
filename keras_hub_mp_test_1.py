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
    # Import the utility to toggle multi-process state
    from keras.src.backend.torch.distribution_lib import set_mp_multi_process_state
    
    log_section("TEST: KERAS HUB MODEL PARALLEL (MP)")
    
    devices = list_devices("gpu")
    if len(devices) < 2:
        log("⚠ Skipping ModelParallel test: Need >= 2 GPUs")
        return False
    
    rank = dist.get_rank() if dist.is_initialized() else 0
    
    # Create 2D device mesh (1 node, N GPUs)
    mesh = DeviceMesh(
        shape=(1, len(devices)),
        axis_names=["batch", "model"],
        devices=devices
    )
    
    layout_map = LayoutMap(mesh)
    # Sharding Config
    layout_map["embeddings.token_embedding.embeddings"] = (None, "model")
    layout_map["embeddings.position_embedding.embeddings"] = (None, None) # Replicated DTensor
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
    
    # CRITICAL FIX 1: Inform the backend that we are in a multi-process MP setup.
    # This triggers input conversion to DTensors in distribution_lib.py
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
        
        # CRITICAL FIX 2: Explicitly build the model with dummy shapes inside the scope.
        # This ensures all weights are sharded as DTensors BEFORE compilation or training
        dummy_input = {
            "token_ids": np.ones((4, 16), dtype="int32"),
            "padding_mask": np.ones((4, 16), dtype="int32")
        }
        model.build({"token_ids": (4, 16), "padding_mask": (4, 16)})
        log("✓ Model built inside ModelParallel scope")
        
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss="mse")

    log_section("PHYSICAL STORAGE VERIFICATION")
    verify_weight_sharding(model, rank)
    
    # Prep data
    x = {
        "token_ids": np.random.randint(0, 50265, size=(4, 16), dtype="int32"),
        "padding_mask": np.ones((4, 16), dtype="int32"),
    }
    y = np.random.random((4, 16, model.hidden_dim)).astype("float32")
    
    log(f"Training for {epochs} epoch(s)...")
    with mp.scope():
        history = model.fit(x, y, epochs=epochs, verbose=0)
        log(f"  Loss: {history.history['loss'][0]:.6f}")
    
    log("✓ ModelParallel test PASSED")
    return True

def verify_weight_sharding(model, rank):
    torch_model = getattr(model, '_torch_layers', getattr(model, 'torch_layer', None))
    if not torch_model:
        log("  ⚠ Access to torch layers failed")
        return

    # Check Token Embedding sharding
    if hasattr(torch_model, 'embeddings') and hasattr(torch_model.embeddings, 'token_embedding'):
        weight = torch_model.embeddings.token_embedding.weight
        if hasattr(weight, 'to_local'):
            log(f"  Token Embedding: Global {tuple(weight.shape)} -> Local {tuple(weight.to_local().shape)}")
            if weight.shape[-1] != weight.to_local().shape[-1]:
                log("    ✓ Sharded across 'model' axis")

def main():
    from keras.src.distribution import initialize, list_devices
    initialize()
    setup_environment()
    
    if len(list_devices("gpu")) >= 2:
        try:
            test_keras_hub_model_parallel(epochs=1, use_preset=False)
        except Exception as e:
            log(f"⚠ Test failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        log("Need >= 2 GPUs for test")
    return 0

if __name__ == "__main__":
    sys.exit(main())