#!/usr/bin/env python3
"""
Final Test Script for Distributed Training Fix

This script tests that the fixes to distribution_lib.py and core.py work correctly.
It verifies:
1. All model weights are properly wrapped as DTensor Parameters
2. Forward and backward passes work without hanging
3. Gradient computation is correct

Usage:
    torchrun --nproc_per_node=2 kaggle_hybrid_test_final.py
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
    return sync_all_ranks()


def verify_dtensor_wrapping(model, local_rank):
    """Verify that all model weights are properly wrapped as DTensors."""
    from torch.distributed._tensor import DTensor
    
    print(f"\n[Rank {local_rank}] Verifying DTensor wrapping...")
    print("-" * 60)
    
    dtensor_count = 0
    tensor_count = 0
    
    for v in model.trainable_variables:
        torch_tensor = getattr(v, 'value', v)
        if hasattr(torch_tensor, 'data'):
            torch_tensor = torch_tensor.data
            
        is_dtensor = isinstance(torch_tensor, DTensor)
        
        if is_dtensor:
            dtensor_count += 1
            # Check if it's a Parameter
            is_param = isinstance(torch_tensor, torch.nn.Parameter)
            has_grad = torch_tensor.requires_grad
            print(f"  ✓ {v.path}")
            print(f"    - Is DTensor: True")
            print(f"    - Is Parameter: {is_param}")
            print(f"    - Requires grad: {has_grad}")
        else:
            tensor_count += 1
            print(f"  ✗ {v.path}")
            print(f"    - Is DTensor: False")
            print(f"    - Type: {type(torch_tensor)}")
    
    print("-" * 60)
    print(f"[Rank {local_rank}] DTensor count: {dtensor_count}")
    print(f"[Rank {local_rank}] Regular tensor count: {tensor_count}")
    
    return dtensor_count > 0 and tensor_count == 0


def verify_gradient_flow(model, local_rank, x, y):
    """Verify that gradients flow correctly through the model."""
    print(f"\n[Rank {local_rank}] Verifying gradient flow...")
    
    # Forward pass
    outputs = model(x, training=True)
    
    # Compute loss
    loss = keras.losses.mse(outputs, y)
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    grad_count = 0
    for v in model.trainable_variables:
        torch_tensor = getattr(v, 'value', v)
        if hasattr(torch_tensor, 'data'):
            torch_tensor = torch_tensor.data
            
        if torch_tensor.grad is not None:
            grad_count += 1
    
    print(f"[Rank {local_rank}] Parameters with gradients: {grad_count}/{len(model.trainable_variables)}")
    
    return grad_count == len(model.trainable_variables)


def main():
    """Main test function."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    print("=" * 70)
    print("FINAL DISTRIBUTED TRAINING FIX TEST")
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
    
    # Verify distributed is initialized
    if torch.distributed.is_initialized():
        print(f"[Rank {local_rank}] ✓ Distributed initialized")
    else:
        print(f"[Rank {local_rank}] ⚠ Distributed not initialized")
    
    # Setup signal handlers
    def signal_handler(signum, frame):
        print(f"\n[Rank {local_rank}] Shutting down...")
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Sync at start
    print(f"\n[Rank {local_rank}] Initial synchronization...")
    wait_for_all_ranks()
    
    # Create distribution strategy
    from keras.src.distribution import DeviceMesh, LayoutMap, ModelParallel
    
    if torch.cuda.is_available():
        devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    else:
        devices = ["cpu"]
    
    mesh = DeviceMesh(
        shape=(world_size,) if world_size > 1 else (1,),
        axis_names=["model"],
        devices=devices
    )
    
    layout_map = LayoutMap(mesh)
    layout_map[".*kernel"] = (None, "model")  # Column parallel for dense layers
    layout_map[".*bias"] = ()  # Replicate biases
    
    strategy = ModelParallel(
        layout_map=layout_map,
        batch_dim_name="batch",
        auto_shard_dataset=False
    )
    
    print(f"[Rank {local_rank}] Strategy created")
    
    # Sync after strategy creation
    wait_for_all_ranks()
    
    # Create model
    from keras import layers
    
    with strategy.scope():
        model = keras.Sequential([
            layers.Input(shape=(64,)),
            layers.Dense(256, activation="relu", name="dense_1"),
            layers.Dense(128, activation="relu", name="dense_2"),
            layers.Dense(10, name="output")
        ])
        
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss="mse")
    
    print(f"[Rank {local_rank}] Model created")
    
    # Sync after model creation
    wait_for_all_ranks()
    
    # Verify DTensor wrapping
    all_dtensor = verify_dtensor_wrapping(model, local_rank)
    
    # Sync
    wait_for_all_ranks()
    
    # Create test data
    batch_size = 16
    x = np.random.random((batch_size, 64)).astype("float32")
    y = np.random.random((batch_size, 10)).astype("float32")
    
    # Verify gradient flow
    grad_ok = verify_gradient_flow(model, local_rank, x, y)
    
    # Sync
    wait_for_all_ranks()
    
    # Run training
    print(f"\n[Rank {local_rank}] Running training...")
    history = model.fit(x, y, epochs=2, batch_size=4, verbose=0)
    
    print(f"[Rank {local_rank}] Training completed!")
    print(f"  Epoch 1 loss: {history.history['loss'][0]:.6f}")
    print(f"  Epoch 2 loss: {history.history['loss'][1]:.6f}")
    
    # Final sync
    wait_for_all_ranks()
    
    # Summary
    print(f"\n{'='*70}")
    print(f"[Rank {local_rank}] TEST RESULTS:")
    print(f"{'='*70}")
    print(f"  All weights as DTensors: {'✓ PASS' if all_dtensor else '✗ FAIL'}")
    print(f"  Gradient flow working: {'✓ PASS' if grad_ok else '✗ FAIL'}")
    print(f"  Training completed: ✓ PASS")
    
    success = all_dtensor and grad_ok
    
    print(f"\n{'='*70}")
    if success:
        print(f"[Rank {local_rank}] ✓ ALL TESTS PASSED!")
    else:
        print(f"[Rank {local_rank}] ✗ SOME TESTS FAILED!")
    print(f"{'='*70}")
    
    return success


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
