#!/usr/bin/env python3
"""
Simple test to verify the mixed tensor fix works for hybrid DP+MP training.

This test runs with torchrun to simulate the multi-process environment.
"""

import os
# Must be set before any other imports
os.environ["KERAS_BACKEND"] = "torch"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np


def run_simple_test():
    """Run a simple test to verify the fix."""
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    print(f"\n{'='*70}")
    print(f"TEST: SIMPLE HYBRID DP+MP (Rank {local_rank}/{world_size})")
    print(f"{'='*70}")
    
    import torch
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        gpu_id = local_rank % num_gpus
        torch.cuda.set_device(gpu_id)
        print(f"Process {local_rank} -> GPU {gpu_id}")
    
    # Initialize distributed backend
    from keras.src.distribution import DeviceMesh, LayoutMap, ModelParallel, initialize
    initialize()
    
    # Create DeviceMesh
    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    mesh = DeviceMesh(
        shape=(len(devices),),
        axis_names=["model"],
        devices=devices
    )
    
    # Create LayoutMap with sharding
    layout_map = LayoutMap(mesh)
    layout_map[".*dense.*kernel"] = (None, "model")  # Shard on dim 1
    layout_map[".*dense.*bias"] = ()  # Replicate
    
    # Create strategy
    strategy = ModelParallel(
        layout_map=layout_map,
        batch_dim_name="data",
        auto_shard_dataset=False
    )
    
    # Build model
    import keras
    from keras import layers
    
    with strategy.scope():
        print(f"\n[Rank {local_rank}] Building model...")
        
        model = keras.Sequential([
            layers.Input(shape=(64,)),
            layers.Dense(256, activation="relu", name="dense_1"),
            layers.Dense(512, activation="relu", name="dense_2"),
            layers.Dense(10, name="output")
        ])
        
        print(f"[Rank {local_rank}] ✓ Model built")
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss='mse',
            metrics=['mae']
        )
    
    # Check sharding
    print(f"\n{'='*70}")
    print(f"MODEL VERIFICATION (Rank {local_rank})")
    print(f"{'='*70}")
    
    from torch.distributed._tensor import DTensor
    
    sharded_count = 0
    replicated_count = 0
    
    for v in model.variables:
        torch_tensor = None
        
        if hasattr(v, 'value') and hasattr(v.value, 'data'):
            torch_tensor = v.value.data
        elif hasattr(v, 'value'):
            torch_tensor = v.value
        else:
            torch_tensor = v
        
        if isinstance(torch_tensor, DTensor):
            local_shape = tuple(torch_tensor.to_local().shape)
            global_shape = tuple(torch_tensor.shape)
            print(f"  {v.path}: {global_shape} -> Local: {local_shape}")
            if local_shape[1] < global_shape[1]:
                print(f"    ✓ SHARDED")
                sharded_count += 1
            else:
                print(f"    - Replicated")
                replicated_count += 1
        elif hasattr(torch_tensor, 'shape'):
            print(f"  {v.path}: {tuple(torch_tensor.shape)} (regular tensor)")
            replicated_count += 1
    
    print(f"\n[Rank {local_rank}] Summary: Sharded={sharded_count}, Replicated={replicated_count}")
    
    # Training test
    print(f"\n{'='*70}")
    print(f"TEST: TRAINING (Rank {local_rank})")
    print(f"{'='*70}")
    
    try:
        # Use numpy arrays for training data
        train_x = np.random.random((32, 64)).astype("float32")
        train_y = np.random.random((32, 10)).astype("float32")
        
        print(f"[Rank {local_rank}] Training data: x={train_x.shape}, y={train_y.shape}")
        
        # The fix should allow training to proceed without mixed tensor errors
        history = model.fit(train_x, train_y, epochs=2, verbose=1, batch_size=8)
        
        print(f"\n[Rank {local_rank}] ✓ Training successful!")
        print(f"Final loss: {history.history['loss'][-1]:.6f}")
        
        # Verify training worked
        initial_loss = history.history['loss'][0]
        final_loss = history.history['loss'][-1]
        improvement = (initial_loss - final_loss) / initial_loss * 100
        print(f"Loss improvement: {improvement:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"[Rank {local_rank}] ✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    import sys
    success = run_simple_test()
    print(f"\n{'='*70}")
    print("TEST COMPLETE")
    print(f"{'='*70}")
    sys.exit(0 if success else 1)

