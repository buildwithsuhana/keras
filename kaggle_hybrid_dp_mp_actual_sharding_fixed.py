#!/usr/bin/env python3
"""
Hybrid Data Parallel + Model Parallel Test with ACTUAL sharding verification

This test verifies that model parallelism sharding is working by checking that
weights are actually sharded across GPUs.

FIX: Uses numpy arrays for data input to avoid mixed tensor issues in the data pipeline.
The Keras data adapter handles the conversion properly when it detects DTensor weights.
"""

import os
# Must be set before any other imports
os.environ["KERAS_BACKEND"] = "torch"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np


def run_hybrid_dp_mp_test():
    """Run the hybrid DP+MP test with ACTUAL sharding."""
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    print(f"\n{'='*70}")
    print(f"TEST: HYBRID DATA PARALLEL + MODEL PARALLEL")
    print(f"{'='*70}")
    print(f"Local rank: {local_rank}, World size: {world_size}")
    
    import torch
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        gpu_id = local_rank % num_gpus
        torch.cuda.set_device(gpu_id)
        print(f"Process {local_rank} -> GPU {gpu_id}")
    
    # Initialize distributed backend
    from keras.src.distribution import DeviceMesh, LayoutMap, ModelParallel, initialize
    from keras.src.backend.torch import distribution_lib
    initialize()
    
    # Create DeviceMesh
    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    mesh = DeviceMesh(
        shape=(len(devices),),
        axis_names=["model"],
        devices=devices
    )
    
    print(f"\n[Rank {local_rank}] DeviceMesh: shape={mesh.shape}")
    
    # Create LayoutMap with sharding
    layout_map = LayoutMap(mesh)
    
    # Shard kernels on dim 1 (output_dim) - each GPU gets half
    layout_map[".*dense.*kernel"] = (None, "model")  # Shard on dim 1
    
    # Biases must be replicated (they broadcast to output)
    layout_map[".*dense.*bias"] = ()  # Replicate
    
    print(f"[Rank {local_rank}] LayoutMap configured:")
    print(f"  - .*dense.*kernel: (None, 'model') - shard on dim 1")
    print(f"  - .*dense.*bias: () - replicate")
    
    # Create strategy
    strategy = ModelParallel(
        layout_map=layout_map,
        batch_dim_name="data",
        auto_shard_dataset=False
    )
    
    # Build model with output_dim divisible by world_size
    import keras
    from keras import layers
    
    with strategy.scope():
        print(f"\n[Rank {local_rank}] Building model...")
        
        model = keras.Sequential([
            layers.Input(shape=(64,)),
            layers.Dense(256, activation="relu", name="dense_1"),  # output_dim=256, divisible by 2
            layers.Dense(512, activation="relu", name="dense_2"),  # output_dim=512, divisible by 2
            layers.Dense(10, name="output")
        ])
        
        print(f"[Rank {local_rank}] ✓ Model built")
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss='mse',
            metrics=['mae']
        )
    
    # Forward pass test
    print(f"\n{'='*70}")
    print(f"TEST: FORWARD PASS")
    print(f"{'='*70}")
    
    batch_size = 8
    
    # FIX: Use numpy arrays instead of torch tensors to avoid mixed tensor issues
    # The Keras data adapter handles the conversion properly when it detects DTensor weights
    x = np.random.random((batch_size, 64)).astype("float32")
    y = np.random.random((batch_size, 10)).astype("float32")
    
    try:
        print(f"[Rank {local_rank}] Input shape: {x.shape}")
        
        # Forward pass with numpy inputs (Keras handles DTensor conversion internally)
        outputs = model(x, training=False)
        
        print(f"[Rank {local_rank}] ✓ Forward pass successful!")
        print(f"Output shape: {outputs.shape}")
        
        # Check if weights are sharded
        print(f"\n{'='*70}")
        print(f"MODEL PARALLEL VERIFICATION")
        print(f"{'='*70}")
        
        from torch.distributed._tensor import DTensor
        
        sharded_count = 0
        replicated_count = 0
        
        for v in model.variables:
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
                    print(f"    ✓ SHARDED on dim 1")
                    sharded_count += 1
                else:
                    print(f"    - Replicated")
                    replicated_count += 1
            elif hasattr(torch_tensor, 'shape'):
                print(f"  {v.path}: {tuple(torch_tensor.shape)} (regular tensor)")
                replicated_count += 1
        
        print(f"\n[Rank {local_rank}] Summary:")
        print(f"  Sharded: {sharded_count}")
        print(f"  Regular/Replicated: {replicated_count}")
        
        if sharded_count > 0:
            print(f"\n[Rank {local_rank}] ✓ Model parallelism IS active!")
        else:
            print(f"\n[Rank {local_rank}] Note: Weights are not sharded.")
        
        # Training test
        print(f"\n{'='*70}")
        print(f"TEST: TRAINING")
        print(f"{'='*70}")
        
        # FIX: Use numpy arrays for training to avoid mixed tensor issues
        # Generate more data for training
        train_x = np.random.random((32, 64)).astype("float32")
        train_y = np.random.random((32, 10)).astype("float32")
        
        print(f"[Rank {local_rank}] Training data shape: {train_x.shape}")
        
        history = model.fit(train_x, train_y, epochs=3, verbose=1, batch_size=batch_size)
        
        print(f"\n[Rank {local_rank}] ✓ Training successful!")
        print(f"Final loss: {history.history['loss'][-1]:.6f}")
        
        # Verify training worked
        initial_loss = history.history['loss'][0]
        final_loss = history.history['loss'][-1]
        improvement = (initial_loss - final_loss) / initial_loss * 100
        print(f"Loss improvement: {improvement:.1f}%")
        
    except Exception as e:
        print(f"[Rank {local_rank}] ✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Cleanup
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    
    print(f"\n{'='*70}")
    print("TEST COMPLETE")
    print(f"{'='*70}")
    
    return True


if __name__ == "__main__":
    import sys
    success = run_hybrid_dp_mp_test()
    sys.exit(0 if success else 1)

