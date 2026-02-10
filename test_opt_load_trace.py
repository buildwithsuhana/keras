#!/usr/bin/env python3
"""
OPT Model Load Hang Tracer

This script traces exactly where the hang occurs during OPT model loading.
"""

import os
# Must be set before any other imports
os.environ["KERAS_BACKEND"] = "torch"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Set NCCL environment variables for better reliability
os.environ["NCCL_BLOCKING_WAIT"] = "0"
os.environ["NCCL_DEBUG"] = "WARN"

import torch
import numpy as np
import signal
import sys
import time

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


def run_opt_load_trace():
    """Trace OPT model loading to find hang location."""
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    print(f"\n{'='*70}")
    print(f"TEST: OPT MODEL LOAD TRACE")
    print(f"{'='*70}")
    print(f"Local rank: {local_rank}, World size: {world_size}")
    
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        gpu_id = local_rank % num_gpus
        torch.cuda.set_device(gpu_id)
        print(f"[Rank {local_rank}] Process -> GPU {gpu_id}")
    
    # Initialize distributed backend
    from keras.src.distribution import DeviceMesh, LayoutMap, ModelParallel, initialize
    from keras.src.backend.torch import distribution_lib
    
    print(f"\n[{local_rank}] Initializing distributed backend...")
    initialize()
    
    if torch.distributed.is_initialized():
        print(f"[{local_rank}] ✓ Distributed backend initialized")
    else:
        print(f"[{local_rank}] ✗ Distributed backend NOT initialized")
    
    # Create DeviceMesh
    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    mesh = DeviceMesh(
        shape=(len(devices),),
        axis_names=["model"],
        devices=devices
    )
    
    print(f"[{local_rank}] DeviceMesh: shape={mesh.shape}")
    
    # Create minimal LayoutMap
    layout_map = LayoutMap(mesh)
    # NO SHARDING - just to test if the issue is with sharding
    layout_map[".*output.*kernel"] = (None, "model")
    
    print(f"[{local_rank}] LayoutMap: Created")
    
    # Create strategy
    strategy = ModelParallel(
        layout_map=layout_map,
        batch_dim_name="data",
        auto_shard_dataset=False
    )
    
    print(f"[{local_rank}] Strategy: Created")
    _sync_cuda()
    
    # ============================================================
    # TRACED MODEL LOAD
    # ============================================================
    
    # Build OPT model
    model = None
    
    print(f"\n[{local_rank}] === STARTING MODEL LOAD ===")
    
    # Test WITHOUT strategy scope first
    print(f"[{local_rank}] Testing WITHOUT strategy scope...")
    _sync_cuda()
    
    try:
        import keras_hub
        
        print(f"[{local_rank}] About to load model...")
        _sync_cuda()
        
        model = keras_hub.models.OPTCausalLM.from_preset("opt_125m_en")
        
        print(f"[{local_rank}] Model loaded (no strategy scope)")
        _sync_cuda()
        
    except Exception as e:
        print(f"[{local_rank}] Load without strategy failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Now test WITH strategy scope
    print(f"\n[{local_rank}] Testing WITH strategy scope...")
    _sync_cuda()
    
    try:
        print(f"[{local_rank}] About to enter strategy scope...")
        _sync_cuda()
        
        with strategy.scope():
            print(f"[{local_rank}] Inside strategy scope")
            _sync_cuda()
            
            print(f"[{local_rank}] About to load model in scope...")
            _sync_cuda()
            
            model2 = keras_hub.models.OPTCausalLM.from_preset("opt_125m_en")
            
            print(f"[{local_rank}] Model loaded (with strategy scope)")
            _sync_cuda()
            
            # Try to access variables
            print(f"[{local_rank}] About to access trainable_variables...")
            _sync_cuda()
            
            vars = model2.trainable_variables
            print(f"[{local_rank}] trainable_variables accessed: {len(vars)}")
            
            _sync_cuda()
            
    except Exception as e:
        print(f"[{local_rank}] Load with strategy failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ============================================================
    # Test forward pass WITHOUT DTensor
    # ============================================================
    
    print(f"\n[{local_rank}] === TESTING FORWARD PASS ===")
    
    try:
        # Create test input
        batch_size = 1
        seq_length = 4
        vocab_size = 50272
        
        token_ids = np.random.randint(0, min(100, vocab_size), size=(batch_size, seq_length))
        padding_mask = np.ones((batch_size, seq_length), dtype=np.int32)
        
        inputs = {"token_ids": token_ids, "padding_mask": padding_mask}
        
        print(f"[{local_rank}] About to call model...")
        _sync_cuda()
        
        with torch.no_grad():
            outputs = model2(inputs)
        
        print(f"[{local_rank}] Forward pass completed!")
        print(f"[{local_rank}] Output shape: {tuple(outputs.shape)}")
        _sync_cuda()
        
    except Exception as e:
        print(f"[{local_rank}] Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ============================================================
    # Test forward pass WITH DTensor
    # ============================================================
    
    print(f"\n[{local_rank}] === TESTING FORWARD PASS WITH DTENSOR ===")
    
    try:
        # Prepare input with distribution
        print(f"[{local_rank}] About to prepare input for distribution...")
        _sync_cuda()
        
        prepared = distribution_lib.prepare_input_for_distribution(inputs)
        
        print(f"[{local_rank}] Input prepared")
        print(f"[{local_rank}] Prepared type: {type(prepared)}")
        _sync_cuda()
        
        # Call model
        print(f"[{local_rank}] About to call model with prepared input...")
        _sync_cuda()
        
        with torch.no_grad():
            outputs_dtensor = model2(prepared)
        
        print(f"[{local_rank}] Forward pass with DTensor completed!")
        print(f"[{local_rank}] Output type: {type(outputs_dtensor)}")
        _sync_cuda()
        
    except Exception as e:
        print(f"[{local_rank}] Forward pass with DTensor failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Sync and finish
    print(f"\n[{local_rank}] Syncing...")
    _safe_barrier()
    
    print(f"\n{'='*70}")
    print(f"TEST COMPLETE - SUCCESS")
    print(f"{'='*70}")
    
    return True


if __name__ == "__main__":
    success = False
    try:
        success = run_opt_load_trace()
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

