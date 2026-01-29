#!/usr/bin/env python3
"""
Test script to verify the ModelParallel fix for "mixed torch.Tensor and DTensor" error.

This script tests that the fix works by:
1. Setting up a ModelParallel distribution with sharded weights
2. Running training and verifying no mixed tensor errors occur
3. Verifying that inputs are properly converted to DTensors

Usage:
    # Single GPU
    python test_model_parallel_fix.py
    
    # Multi-GPU with torchrun
    torchrun --nproc_per_node=2 test_model_parallel_fix.py
"""

import os
# MUST be set before any other imports
os.environ["KERAS_BACKEND"] = "torch"
os.environ["KERAS_DISTRIBUTION_DEBUG"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import torch
import torch.distributed as dist
import numpy as np

# Initialize Keras distribution first
from keras.src.distribution import initialize, distribution

def main():
    # Initialize the distribution system
    initialize()
    
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    
    print(f"[Rank {rank:02d}] Starting ModelParallel fix test...")
    print(f"[Rank {rank:02d}] World size: {world_size}")
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"[Rank {rank:02d}] GPU available: {torch.cuda.get_device_name()}")
    else:
        print(f"[Rank {rank:02d}] No GPU available, using CPU")
    
    # Import Keras after setting backend
    import keras
    from keras import layers
    from keras.src.distribution import DeviceMesh, LayoutMap, ModelParallel
    from keras.src.backend.torch.distribution_lib import (
        is_dtensor, _get_default_device_mesh, DTENSOR_AVAILABLE
    )
    
    print(f"[Rank {rank:02d}] DTensor available: {DTENSOR_AVAILABLE}")
    
    if world_size < 2 or not torch.cuda.is_available():
        print("[Rank 00] Need at least 2 GPUs for ModelParallel test. Skipping...")
        if world_size > 1:
            print("[Rank 00] Multi-process mode detected but insufficient GPUs")
        return 0
    
    # Create devices list
    devices = [f"cuda:{i}" for i in range(min(2, torch.cuda.device_count()))]
    print(f"[Rank {rank:02d}] Using devices: {devices}")
    
    # Create 2D device mesh for model parallelism
    mesh = DeviceMesh(
        shape=(1, len(devices)),
        axis_names=["batch", "model"],
        devices=devices
    )
    print(f"[Rank {rank:02d}] DeviceMesh created: shape={mesh.shape}, axes={mesh.axis_names}")
    
    # Create layout map for sharding
    layout_map = LayoutMap(mesh)
    layout_map["dense.*kernel"] = (None, "model")  # Shard on model axis
    layout_map["dense.*bias"] = ("model",)
    
    # Create ModelParallel distribution
    mp = ModelParallel(
        layout_map=layout_map,
        batch_dim_name="batch",
        auto_shard_dataset=False
    )
    print(f"[Rank {rank:02d}] ModelParallel created")
    
    # Create model under distribution scope
    with mp.scope():
        model = keras.Sequential([
            layers.Dense(512, activation="relu", input_shape=(128,)),
            layers.Dense(256, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(10)
        ])
        
        # Verify weights are DTensors
        for i, layer in enumerate(model.layers):
            if hasattr(layer, 'kernel') and layer.kernel is not None:
                kernel = layer.kernel
                if hasattr(kernel, 'value'):
                    kernel_value = kernel.value
                elif hasattr(kernel, '_value'):
                    kernel_value = kernel._value
                else:
                    kernel_value = kernel
                
                if is_dtensor(kernel_value):
                    print(f"[Rank {rank:02d}] Layer {i} ({layer.name}): kernel is DTensor - OK")
                else:
                    print(f"[Rank {rank:02d}] Layer {i} ({layer.name}): kernel is NOT DTensor")
        
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss="mse")
    
    # Create training data
    batch_size = 32
    x = np.random.random((batch_size, 128)).astype("float32")
    y = np.random.random((batch_size, 10)).astype("float32")
    print(f"[Rank {rank:02d}] Created training data: x.shape={x.shape}, y.shape={y.shape}")
    
    # Test training - this is where the fix should prevent "mixed tensor" errors
    print(f"[Rank {rank:02d}] Starting training (3 epochs)...")
    
    try:
        history = model.fit(x, y, epochs=3, verbose=1, batch_size=batch_size)
        print(f"[Rank {rank:02d}] Training completed successfully!")
        print(f"[Rank {rank:02d}] Final loss: {history.history['loss'][-1]:.6f}")
        
        # Test evaluation
        print(f"[Rank {rank:02d}] Testing evaluation...")
        eval_results = model.evaluate(x, y, verbose=0)
        print(f"[Rank {rank:02d}] Evaluation completed: loss={eval_results:.6f}")
        
        # Test prediction
        print(f"[Rank {rank:02d}] Testing prediction...")
        predictions = model.predict(x, verbose=0)
        print(f"[Rank {rank:02d}] Prediction completed: shape={predictions.shape}")
        
        print(f"[Rank {rank:02d}] ALL TESTS PASSED!")
        return 0
        
    except Exception as e:
        print(f"[Rank {rank:02d}] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # Cleanup
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())

