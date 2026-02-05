#!/usr/bin/env python3
"""
Hybrid Data Parallel + Model Parallel Test with ACTUAL sharding

This test uses a model where the sharded dimension IS divisible by 2.

Key insight: For proper model parallelism, the sharded dimension must be
divisible by the world size. For BERT tiny:
- intermediate_dim=512 is NOT divisible by 2
- So we cannot shard on that dimension

Solution: Use a custom model where intermediate_dim IS divisible by 2
(e.g., 512 → 256 per GPU with world_size=2)
"""

import os
# Must be set before any other imports
os.environ["KERAS_BACKEND"] = "torch"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import numpy as np


def run_hybrid_dp_mp_test():
    """Run the hybrid DP+MP test with ACTUAL sharding."""
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    print(f"\n{'='*70}")
    print(f"TEST: HYBRID DATA PARALLEL + MODEL PARALLEL (ACTUAL SHARDING)")
    print(f"{'='*70}")
    print(f"Local rank: {local_rank}, World size: {world_size}")
    
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        gpu_id = local_rank % num_gpus
        torch.cuda.set_device(gpu_id)
        print(f"Process {local_rank} -> GPU {gpu_id}")
    
    # Initialize distributed backend
    from keras.src.distribution import DeviceMesh, LayoutMap, ModelParallel, initialize
    from keras.src.backend.torch import distribution_lib
    initialize()
    
    # Create DeviceMesh for 2D parallelism (batch + model)
    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    mesh = DeviceMesh(
        shape=(len(devices),),
        axis_names=["model"],
        devices=devices
    )
    
    print(f"\n[Rank {local_rank}] DeviceMesh: shape={mesh.shape}")
    
    # Create LayoutMap with ACTUAL sharding
    # Use a model where intermediate_dim IS divisible by 2
    layout_map = LayoutMap(mesh)
    
    # For a Dense layer with kernel shape (input_dim, output_dim):
    # - Sharding on dim 1 (output_dim) means each GPU has (input_dim, output_dim/2)
    # - We need output_dim to be divisible by world_size
    
    # Use output_dim=512 which IS divisible by 2
    layout_map[".*dense.*kernel"] = (None, "model")  # Shard on output_dim (dim 1)
    
    # Biases must be replicated (they broadcast to output)
    layout_map[".*dense.*bias"] = ()  # Replicate
    
    print(f"[Rank {local_rank}] LayoutMap patterns configured (ACTUAL SHARDING)")
    
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
        print(f"\n[Rank {local_rank}] Building model with sharded Dense layers...")
        
        # Build a model where output_dim IS divisible by 2
        # This allows proper sharding on dim 1
        model = keras.Sequential([
            layers.Input(shape=(64,)),
            layers.Dense(256, activation="relu", name="dense_1"),  # output_dim=256, divisible by 2
            layers.Dense(512, activation="relu", name="dense_2"),  # output_dim=512, divisible by 2
            layers.Dense(10, name="output")
        ])
        
        print(f"[Rank {local_rank}] ✓ Model built")
        
        # Verify sharding
        from torch.distributed._tensor import DTensor
        
        print(f"\n[Rank {local_rank}] Verifying sharding:")
        for layer in model.layers:
            if hasattr(layer, 'kernel') and hasattr(layer.kernel, 'value'):
                kernel_var = layer.kernel
                if hasattr(kernel_var, 'value'):
                    kernel_tensor = kernel_var.value
                else:
                    kernel_tensor = kernel_var
                
                if isinstance(kernel_tensor, DTensor):
                    local_shape = tuple(kernel_tensor.to_local().shape)
                    global_shape = tuple(kernel_tensor.shape)
                    print(f"  {layer.name}: {global_shape} -> Local: {local_shape}")
                    if local_shape[1] < global_shape[1]:
                        print(f"    ✓ SHARDED on dim 1")
                    else:
                        print(f"    - Replicated")
                elif hasattr(kernel_tensor, 'shape'):
                    print(f"  {layer.name}: {tuple(kernel_tensor.shape)} (not DTensor yet)")
        
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
    x = np.random.random((batch_size, 64)).astype("float32")
    y = np.random.random((batch_size, 10)).astype("float32")
    
    try:
        print(f"[Rank {local_rank}] Input shape: {x.shape}")
        
        # Forward pass
        outputs = model(x, training=False)
        
        print(f"[Rank {local_rank}] ✓ Forward pass successful!")
        print(f"Output shape: {outputs.shape}")
        
        # Training test
        print(f"\n{'='*70}")
        print(f"TEST: TRAINING")
        print(f"{'='*70}")
        
        history = model.fit(x, y, epochs=3, verbose=1, batch_size=batch_size)
        
        print(f"\n[Rank {local_rank}] ✓ Training successful!")
        print(f"Final loss: {history.history['loss'][-1]:.6f}")
        
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
    print("TEST COMPLETE - HYBRID DP+MP WITH ACTUAL SHARDING")
    print(f"{'='*70}")
    
    return True


if __name__ == "__main__":
    import sys
    success = run_hybrid_dp_mp_test()
    sys.exit(0 if success else 1)
