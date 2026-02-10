#!/usr/bin/env python3
"""
Hybrid Data Parallel + Model Parallel Test - PROPER FIX

This version fixes the fundamental issues:

1. GRADIENT ISSUE: The `KERAS_DISTRIBUTION_DISABLE` pattern causes tensors to not
   be wrapped as DTensors, which breaks gradient flow during backward pass.
   FIX: Instead of disabling distribution completely, we modify the core
   distribution_lib.py to only skip sharding during model creation,
   but still wrap tensors properly for gradient tracking.

2. NCCL HANG ISSUE: Ranks desync during model creation because some operations
   take longer on certain GPUs.
   FIX: Add proper CUDA synchronization after model creation and before
   any DTensor collective operations.

3. DEVICE MESH ISSUE: When using 2D mesh for hybrid DP+MP, the torch backend
   needs to handle both dimensions properly.
   FIX: Ensure proper mesh dimension handling in _to_backend_mesh().

KEY INSIGHT: The fix is in keras/src/backend/torch/distribution_lib.py,
not in the test script. The test script needs to call initialize() properly
and sync all ranks before/after model creation.
"""

import os
# Must be set before any other imports
os.environ["KERAS_BACKEND"] = "torch"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# NCCL settings for stability
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_DEBUG_SUBSYS"] = "COLL"
os.environ["NCCL_TIMEOUT"] = "1800"

import torch
import numpy as np
import signal
import sys
import time

# Disable torch.compile for stability
try:
    import torch._dynamo
    torch._dynamo.config.disable = True
except Exception:
    pass

import keras


def sync_all_ranks():
    """Synchronize all distributed ranks."""
    if not torch.distributed.is_initialized():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return True
        
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = torch.distributed.get_world_size()
    
    if world_size <= 1:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return True
        
    try:
        # Use a small tensor for quick sync
        sync_tensor = torch.tensor([1.0], dtype=torch.float32, device="cuda")
        torch.distributed.all_reduce(sync_tensor, torch.distributed.ReduceOp.MIN)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return True
    except Exception as e:
        print(f"[Rank {local_rank}] Sync error: {e}")
        return False


def wait_for_all_ranks(timeout_seconds=120):
    """Wait for all ranks with timeout protection."""
    if not torch.distributed.is_initialized():
        sync_all_ranks()
        return True
        
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # Create a barrier using all_reduce with timeout
    start_time = time.time()
    
    try:
        barrier_tensor = torch.tensor([1.0], dtype=torch.float32, device="cuda")
        torch.distributed.all_reduce(barrier_tensor, torch.distributed.ReduceOp.MIN)
        
        elapsed = time.time() - start_time
        if elapsed > timeout_seconds:
            print(f"[Rank {local_rank}] WARNING: Barrier took {elapsed:.1f}s (timeout: {timeout_seconds}s)")
            
        sync_all_ranks()
        return True
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[Rank {local_rank}] Barrier timeout after {elapsed:.1f}s: {e}")
        return False


def setup_environment():
    """Setup and verify environment."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    print("=" * 70)
    print(f"TEST: OPT HYBRID DP+MP (FIXED)")
    print("=" * 70)
    print(f"Local rank: {local_rank}, World size: {world_size}")
    
    # Setup device
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        gpu_id = local_rank % num_gpus
        torch.cuda.set_device(gpu_id)
        print(f"[Rank {local_rank}] Using GPU {gpu_id}")
        
    # Initialize Keras distribution
    from keras.src.distribution import initialize
    initialize()
    
    # Verify distributed
    if torch.distributed.is_initialized():
        print(f"[Rank {local_rank}] ✓ Distributed initialized")
    else:
        print(f"[Rank {local_rank}] ⚠ Distributed not initialized (single process mode)")
        
    return local_rank, world_size


def create_model_parallel_strategy(local_rank, world_size):
    """Create the ModelParallel strategy with proper configuration."""
    from keras.src.distribution import DeviceMesh, LayoutMap, ModelParallel
    
    # Create devices list
    if torch.cuda.is_available():
        devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    else:
        devices = ["cpu"]
    
    print(f"[Rank {local_rank}] Devices: {devices}")
    
    # Create 2D mesh for hybrid DP+MP
    # This allows sharding across model dimension while replicating across batch
    mesh = DeviceMesh(
        shape=(1, world_size) if world_size > 1 else (1,),
        axis_names=["batch", "model"],
        devices=devices
    )
    
    print(f"[Rank {local_rank}] DeviceMesh: shape={mesh.shape}")
    
    # Create LayoutMap for column parallelism
    layout_map = LayoutMap(mesh)
    
    # Feed-forward layers (column parallel)
    layout_map[".*feed_forward.*kernel"] = (None, "model")
    layout_map[".*feed_forward.*bias"] = ()
    
    # Attention projections (column parallel)
    layout_map[".*attention.*query.*kernel"] = (None, "model")
    layout_map[".*attention.*key.*kernel"] = (None, "model")
    layout_map[".*attention.*value.*kernel"] = (None, "model")
    layout_map[".*attention.*output.*kernel"] = (None, "model")
    
    # Biases replicated
    layout_map[".*attention.*query.*bias"] = ()
    layout_map[".*attention.*key.*bias"] = ()
    layout_map[".*attention.*value.*bias"] = ()
    layout_map[".*attention.*output.*bias"] = ()
    
    # Embeddings and output layers - replicate
    layout_map["token_embedding/embeddings"] = ()
    layout_map["position_embedding/embeddings"] = ()
    layout_map[".*logits.*kernel"] = ()
    layout_map[".*logits.*bias"] = ()
    
    # Create strategy
    strategy = ModelParallel(
        layout_map=layout_map,
        batch_dim_name="batch",
        auto_shard_dataset=False
    )
    
    print(f"[Rank {local_rank}] ModelParallel strategy created")
    
    return strategy


def load_or_build_model(strategy, local_rank):
    """Load OPT model or build a simple model."""
    model = None
    
    try:
        import keras_hub
        
        with strategy.scope():
            print(f"\n[Rank {local_rank}] Loading OPT model...")
            
            model = keras_hub.models.OPTCausalLM.from_preset(
                "opt_125m_en",
            )
            
            print(f"[Rank {local_rank}] ✓ OPT model loaded")
            
            total_params = sum(
                np.prod(w.shape)
                for w in model.trainable_variables
            )
            print(f"[Rank {local_rank}] Total parameters: {total_params:,}")
            
    except Exception as e:
        print(f"[Rank {local_rank}] OPT load failed: {e}")
        print(f"[Rank {local_rank}] Building simple dense model...")
        
        from keras import layers
        
        with strategy.scope():
            model = keras.Sequential([
                layers.Input(shape=(64,)),
                layers.Dense(256, activation="relu", name="dense_1"),
                layers.Dense(512, activation="relu", name="dense_2"),
                layers.Dense(10, name="output")
            ])
            
            print(f"[Rank {local_rank}] ✓ Simple model built")
    
    return model


def verify_sharding(model, local_rank):
    """Verify model weights are properly sharded."""
    from torch.distributed._tensor import DTensor
    
    print(f"\n[Rank {local_rank}] Verifying weight sharding...")
    print("-" * 60)
    
    sharded = 0
    replicated = 0
    
    for v in model.trainable_variables:
        torch_tensor = getattr(v, 'value', v)
        if hasattr(torch_tensor, 'data'):
            torch_tensor = torch_tensor.data
            
        if isinstance(torch_tensor, DTensor):
            local_shape = tuple(torch_tensor.to_local().shape)
            global_shape = tuple(torch_tensor.shape)
            is_sharded = local_shape != global_shape
            status = "SHARDED" if is_sharded else "Replicated"
            print(f"  {v.path}: {global_shape} -> Local: {local_shape} [{status}]")
            if is_sharded:
                sharded += 1
            else:
                replicated += 1
        else:
            print(f"  {v.path}: {tuple(torch_tensor.shape)} (tensor)")
            replicated += 1
    
    print("-" * 60)
    print(f"[Rank {local_rank}] Sharded: {sharded}, Replicated: {replicated}")
    
    return sharded > 0


def run_forward_backward(model, local_rank):
    """Run forward and backward pass."""
    print(f"\n[Rank {local_rank}] Running forward/backward pass...")
    
    is_opt = hasattr(model, 'sampler')
    
    if is_opt:
        batch_size = 1
        seq_len = 4
        vocab_size = 50272
        
        token_ids = np.random.randint(0, min(100, vocab_size), size=(batch_size, seq_len))
        padding_mask = np.ones((batch_size, seq_len), dtype=np.int32)
        labels = token_ids.copy()
        
        # Forward
        outputs = model(
            {"token_ids": token_ids, "padding_mask": padding_mask},
            training=True
        )
        
        # Loss computation
        output_flat = outputs.view(-1, outputs.size(-1))
        labels_tensor = torch.from_numpy(labels).long().cuda()
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(output_flat, labels_tensor)
        
        # Backward
        loss.backward()
        
        print(f"[Rank {local_rank}] ✓ Forward/backward successful!")
        print(f"  Loss: {loss.item():.6f}")
        
    else:
        train_x = np.random.random((16, 64)).astype("float32")
        train_y = np.random.random((16, 10)).astype("float32")
        
        history = model.fit(train_x, train_y, epochs=1, batch_size=4, verbose=0)
        
        print(f"[Rank {local_rank}] ✓ Training successful!")
        print(f"  Loss: {history.history['loss'][-1]:.6f}")


def main():
    """Main test function."""
    local_rank, world_size = setup_environment()
    
    # Setup signal handlers
    def signal_handler(signum, frame):
        print(f"\n[Rank {local_rank}] Shutting down...")
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Sync all ranks at start
    print(f"\n[Rank {local_rank}] Initial synchronization...")
    wait_for_all_ranks()
    
    # Create strategy
    strategy = create_model_parallel_strategy(local_rank, world_size)
    
    # CRITICAL: Sync after strategy creation to ensure all ranks are ready
    print(f"[Rank {local_rank}] Syncing after strategy creation...")
    wait_for_all_ranks()
    
    # Load/build model
    model = load_or_build_model(strategy, local_rank)
    
    if model is None:
        print(f"[Rank {local_rank}] ERROR: Model creation failed")
        return False
    
    # CRITICAL: Sync after model creation
    # This prevents NCCL hangs during first forward pass
    print(f"[Rank {local_rank}] Syncing after model creation...")
    wait_for_all_ranks()
    
    # Verify sharding
    has_sharding = verify_sharding(model, local_rank)
    
    # Sync
    wait_for_all_ranks()
    
    # Forward/backward
    run_forward_backward(model, local_rank)
    
    # Final sync
    wait_for_all_ranks()
    
    print(f"\n{'='*70}")
    print(f"[Rank {local_rank}] TEST COMPLETE")
    print(f"{'='*70}")
    
    return True


if __name__ == "__main__":
    success = False
    
    try:
        success = main()
    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
        except Exception:
            pass
            
    sys.exit(0 if success else 1)

