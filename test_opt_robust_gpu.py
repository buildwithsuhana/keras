#!/usr/bin/env python3
"""
OPT Model Load with Robust GPU Discovery

This script handles the Kaggle GPU device discovery issue where GPU IDs
may not match LOCAL_RANK directly.
"""

import os
# Must be set BEFORE any other imports
os.environ["KERAS_BACKEND"] = "torch"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Set NCCL environment variables
os.environ["NCCL_BLOCKING_WAIT"] = "0"
os.environ["NCCL_DEBUG"] = "WARN"

import torch
import numpy as np
import signal
import sys
import time

# Get rank info
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))

print(f"\n{'='*70}")
print(f"TEST: OPT MODEL WITH ROBUST GPU DISCOVERY")
print(f"{'='*70}")
print(f"Local rank: {LOCAL_RANK}, World size: {WORLD_SIZE}")

# DISCOVER GPU DEVICES - This is critical for Kaggle environments
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"[Rank {LOCAL_RANK}] Available GPUs: {num_gpus}")
    
    # Try to discover actual GPU device IDs
    # In some environments, LOCAL_RANK doesn't map directly to GPU IDs
    gpu_ids = list(range(num_gpus))
    print(f"[Rank {LOCAL_RANK}] GPU IDs to try: {gpu_ids}")
    
    # Map LOCAL_RANK to a GPU device
    # Each process needs its own GPU
    gpu_id = gpu_ids[LOCAL_RANK % num_gpus]
    
    # Set device BEFORE any torch.cuda calls
    torch.cuda.set_device(gpu_id)
    print(f"[Rank {LOCAL_RANK}] Set CUDA device to: {gpu_id}")
    print(f"[Rank {LOCAL_RANK}] Current device: {torch.cuda.current_device()}")
    print(f"[Rank {LOCAL_RANK}] GPU device name: {torch.cuda.get_device_name(gpu_id)}")
else:
    print(f"[Rank {LOCAL_RANK}] CUDA not available!")

# Global flag for graceful shutdown
shutdown_requested = False


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global shutdown_requested
    rank = os.environ.get("LOCAL_RANK", "?")
    print(f"\n[Rank {rank}] Received signal {signum}, shutting down gracefully...")
    shutdown_requested = True


# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def _sync_cuda():
    """Synchronize CUDA streams."""
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass


def _safe_barrier():
    """Perform a barrier with error handling."""
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
            _sync_cuda()
    except Exception as e:
        rank = os.environ.get("LOCAL_RANK", "?")
        print(f"[Rank {rank}] Barrier warning: {e}")
        _sync_cuda()


def run_opt_test():
    """Run OPT model test with proper GPU assignment."""
    
    rank = LOCAL_RANK
    
    # Initialize distributed backend AFTER setting CUDA device
    from keras.src.distribution import DeviceMesh, LayoutMap, ModelParallel, initialize
    from keras.src.backend.torch import distribution_lib
    
    print(f"\n[{rank}] Initializing distributed backend...")
    initialize()
    
    if torch.distributed.is_initialized():
        print(f"[{rank}] ✓ Distributed backend initialized")
        print(f"[{rank}] Rank: {torch.distributed.get_rank()}")
        print(f"[{rank}] World size: {torch.distributed.get_world_size()}")
    else:
        print(f"[{rank}] ✗ Distributed backend NOT initialized")
        return False
    
    # Verify each rank has different GPU
    current_device = torch.cuda.current_device()
    print(f"[{rank}] Current CUDA device: {current_device}")
    print(f"[{rank}] GPU: {torch.cuda.get_device_name(current_device)}")
    
    # Sync and verify all ranks can see each other
    print(f"[{rank}] Testing barrier sync...")
    _safe_barrier()
    print(f"[{rank}] Barrier sync OK")
    
    # Create DeviceMesh with proper device mapping
    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    mesh = DeviceMesh(
        shape=(len(devices),),
        axis_names=["model"],
        devices=devices
    )
    
    print(f"[{rank}] DeviceMesh: shape={mesh.shape}, devices={devices}")
    
    # Create minimal LayoutMap - use empty to avoid any sharding first
    layout_map = LayoutMap(mesh)
    # NO SHARDING for initial test - just to verify basic functionality
    layout_map[".*"] = ()  # Replicate all weights
    
    print(f"[{rank}] LayoutMap: Created (all replicated)")
    
    # Create strategy
    strategy = ModelParallel(
        layout_map=layout_map,
        batch_dim_name="data",
        auto_shard_dataset=False
    )
    
    print(f"[{rank}] Strategy: Created")
    _sync_cuda()
    
    # ============================================================
    # TRACED MODEL LOAD
    # ============================================================
    
    print(f"\n[{rank}] === STARTING MODEL LOAD ===")
    
    model = None
    
    try:
        import keras_hub
        
        print(f"[{rank}] About to load model in strategy scope...")
        _sync_cuda()
        
        with strategy.scope():
            print(f"[{rank}] Inside strategy scope")
            _sync_cuda()
            
            model = keras_hub.models.OPTCausalLM.from_preset("opt_125m_en")
            
            print(f"[{rank}] ✓ Model loaded in strategy scope")
            _sync_cuda()
            
            # Access variables
            vars = model.trainable_variables
            print(f"[{rank}] ✓ trainable_variables accessed: {len(vars)}")
            _sync_cuda()
            
    except Exception as e:
        print(f"[{rank}] ✗ Load failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ============================================================
    # Test forward pass
    # ============================================================
    
    print(f"\n[{rank}] === TESTING FORWARD PASS ===")
    
    try:
        # Create test input
        batch_size = 1
        seq_length = 4
        vocab_size = 50272
        
        token_ids = np.random.randint(0, min(100, vocab_size), size=(batch_size, seq_length))
        padding_mask = np.ones((batch_size, seq_length), dtype=np.int32)
        
        inputs = {"token_ids": token_ids, "padding_mask": padding_mask}
        
        # Prepare with distribution
        print(f"[{rank}] Preparing input for distribution...")
        _sync_cuda()
        
        prepared = distribution_lib.prepare_input_for_distribution(inputs)
        print(f"[{rank}] ✓ Input prepared: {type(prepared)}")
        _sync_cuda()
        
        # Forward pass
        print(f"[{rank}] Running forward pass...")
        _sync_cuda()
        
        with torch.no_grad():
            outputs = model(prepared)
        
        print(f"[{rank}] ✓ Forward pass completed!")
        print(f"[{rank}] Output type: {type(outputs)}")
        _sync_cuda()
        
    except Exception as e:
        print(f"[{rank}] ✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Sync and finish
    print(f"\n[{rank}] Syncing...")
    _safe_barrier()
    
    print(f"\n{'='*70}")
    print(f"TEST COMPLETE - SUCCESS (Rank {rank})")
    print(f"{'='*70}")
    
    return True


if __name__ == "__main__":
    success = False
    try:
        success = run_opt_test()
    except KeyboardInterrupt:
        rank = os.environ.get("LOCAL_RANK", "0")
        print(f"\n[Rank {rank}] Interrupted by user")
    except Exception as e:
        rank = os.environ.get("LOCAL_RANK", "0")
        print(f"\n[Rank {rank}] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
                print(f"[Cleanup] Process group destroyed")
        except:
            pass
    
    sys.exit(0 if success else 1)

