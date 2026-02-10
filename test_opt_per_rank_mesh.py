#!/usr/bin/env python3
"""
OPT Model with Proper DeviceMesh per Rank

This script fixes the NCCL "Duplicate GPU detected" error by ensuring
each rank creates a DeviceMesh with only its LOCAL GPU device.
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

# Get rank info BEFORE any CUDA initialization
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))

print(f"\n{'='*70}")
print(f"TEST: OPT MODEL WITH PER-RANK DEVICEMESH")
print(f"{'='*70}")
print(f"Local rank: {LOCAL_RANK}, World size: {WORLD_SIZE}")

# CRITICAL: Set GPU device BEFORE any CUDA operations
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"[Rank {LOCAL_RANK}] Available GPUs: {num_gpus}")
    
    # Each rank gets ONE dedicated GPU
    gpu_id = LOCAL_RANK % num_gpus
    torch.cuda.set_device(gpu_id)
    print(f"[Rank {LOCAL_RANK}] Set CUDA device to: {gpu_id}")
    print(f"[Rank {LOCAL_RANK}] GPU: {torch.cuda.get_device_name(gpu_id)}")

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
    """Run OPT model test with per-rank DeviceMesh."""
    
    rank = LOCAL_RANK
    
    # Initialize distributed backend AFTER setting CUDA device
    from keras.src.distribution import DeviceMesh, LayoutMap, ModelParallel, initialize
    
    print(f"\n[{rank}] Initializing distributed backend...")
    initialize()
    
    if torch.distributed.is_initialized():
        print(f"[{rank}] ✓ Distributed backend initialized")
        print(f"[{rank}] Rank: {torch.distributed.get_rank()}")
        print(f"[{rank}] World size: {torch.distributed.get_world_size()}")
    else:
        print(f"[{rank}] ✗ Distributed backend NOT initialized")
        return False
    
    # Verify GPU assignment
    current_device = torch.cuda.current_device()
    print(f"[{rank}] Current CUDA device: {current_device}")
    print(f"[{rank}] GPU: {torch.cuda.get_device_name(current_device)}")
    
    # CRITICAL FIX: Create DeviceMesh with only the LOCAL device per rank
    # Each rank should only have its own GPU in the mesh
    # The mesh shape should reflect that each rank has 1 local device
    local_device = f"cuda:{torch.cuda.current_device()}"
    
    # For 2-GPU model parallel with 2 ranks, each rank has 1 local device
    # Shape should be (1,) for each rank, and PyTorch DTensor will handle
    # the global mesh across processes
    mesh = DeviceMesh(
        shape=(1,),  # Each rank has 1 local device
        axis_names=["model"],
        devices=[local_device]  # Only the local device
    )
    
    print(f"[{rank}] DeviceMesh: shape={mesh.shape}, devices={mesh.devices}")
    
    # Create LayoutMap - replicate embeddings, only shard output
    layout_map = LayoutMap(mesh)
    layout_map["token_embedding.*"] = ()  # Replicate
    layout_map["position_embedding.*"] = ()  # Replicate
    layout_map[".*output.*kernel"] = ("model",)  # Shard output
    layout_map[".*output.*bias"] = ()  # Replicate
    
    print(f"[{rank}] LayoutMap: Created")
    
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
            print("hello world")
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
        
        # CRITICAL: Prepare inputs for distribution
        # This converts inputs to DTensor when model has DTensor weights
        from keras.src.backend.torch import distribution_lib
        print(f"[{rank}] Preparing inputs for distribution...")
        prepared_inputs = distribution_lib.prepare_input_for_distribution(inputs)
        print(f"[{rank}] Inputs prepared: {type(prepared_inputs)}")
        _sync_cuda()
        
        # Forward pass WITH distribution
        print(f"[{rank}] Running forward pass (with distribution)...")
        
        with torch.no_grad():
            outputs = model(prepared_inputs)
        
        print(f"[{rank}] ✓ Forward pass (with distribution) completed!")
        print(f"[{rank}] Output shape: {tuple(outputs.shape)}")
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

