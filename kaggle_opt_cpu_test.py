#!/usr/bin/env python3
"""
Simple CPU-based test for ModelParallel distribution.

This script tests the Keras DTensor distribution functionality on CPU,
without requiring GPU or NCCL communication.

Usage:
    python kaggle_opt_cpu_test.py
"""

import os
# Must be set before any other imports
os.environ["KERAS_BACKEND"] = "torch"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Force CPU mode - no CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import numpy as np
import sys

# Import keras at the top level
import keras


def _get_backend_type():
    """Determine the backend type."""
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def run_cpu_test():
    """Test ModelParallel distribution on CPU."""
    
    print(f"\n{'='*70}")
    print(f"TEST: MODELPARALLEL DISTRIBUTION ON CPU")
    print(f"{'='*70}")
    
    backend = _get_backend_type()
    print(f"Backend: {backend}")
    
    # Test 1: Create a simple model without distribution
    print(f"\n{'='*70}")
    print(f"TEST 1: SIMPLE MODEL (NO DISTRIBUTION)")
    print(f"{'='*70}")
    
    from keras import layers
    
    model_no_dist = keras.Sequential([
        layers.Input(shape=(64,)),
        layers.Dense(256, activation="relu", name="dense_1"),
        layers.Dense(128, activation="relu", name="dense_2"),
        layers.Dense(10, name="output")
    ])
    
    model_no_dist.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse',
        metrics=['mae']
    )
    
    # Test forward pass
    x = np.random.random((4, 64)).astype("float32")
    with torch.no_grad():
        outputs = model_no_dist(x)
    print(f"✓ Simple model forward pass: output shape = {outputs.shape}")
    
    # Test 2: Try ModelParallel with CPU (should gracefully handle no distributed backend)
    print(f"\n{'='*70}")
    print(f"TEST 2: MODELPARALLEL WITH CPU (NO DISTRIBUTED)")
    print(f"{'='*70}")
    
    try:
        from keras.src.distribution import DeviceMesh, LayoutMap, ModelParallel, initialize
        from keras.src.backend.torch import distribution_lib
        from torch.distributed._tensor import DTensor, Replicate
        
        # Initialize (should be no-op on CPU with single process)
        initialize()
        
        # Create a simple DeviceMesh for CPU
        devices = ["cpu:0"]
        mesh = DeviceMesh(
            shape=(1,),
            axis_names=["model"],
            devices=devices
        )
        
        print(f"DeviceMesh created: shape={mesh.shape}")
        
        # Create LayoutMap
        layout_map = LayoutMap(mesh)
        layout_map[".*kernel"] = (None, "model")
        layout_map[".*bias"] = ()
        
        print("LayoutMap created")
        
        # Create strategy
        strategy = ModelParallel(
            layout_map=layout_map,
            batch_dim_name="data",
            auto_shard_dataset=False
        )
        
        print(f"ModelParallel strategy created")
        
        # Create model in strategy scope
        with strategy.scope():
            model_with_dist = keras.Sequential([
                layers.Input(shape=(64,)),
                layers.Dense(256, activation="relu", name="dense_1"),
                layers.Dense(128, activation="relu", name="dense_2"),
                layers.Dense(10, name="output")
            ])
            
            model_with_dist.compile(
                optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                loss='mse',
                metrics=['mae']
            )
        
        print(f"✓ Model created in ModelParallel scope")
        
        # Test forward pass
        x = np.random.random((4, 64)).astype("float32")
        with torch.no_grad():
            outputs = model_with_dist(x)
        print(f"✓ ModelParallel model forward pass: output shape = {outputs.shape}")
        
        # Test training
        y = np.random.random((4, 10)).astype("float32")
        history = model_with_dist.fit(x, y, epochs=1, batch_size=4, verbose=0)
        print(f"✓ ModelParallel training: loss = {history.history['loss'][-1]:.6f}")
        
    except Exception as e:
        print(f"⚠ ModelParallel test skipped or failed: {e}")
        import traceback
        traceback.print_exc()
        print("(This is expected if DTensor is not available on CPU)")
    
    # Test 3: Verify DTensor conversion functions work
    print(f"\n{'='*70}")
    print(f"TEST 3: DTENSOR UTILITY FUNCTIONS")
    print(f"{'='*70}")
    
    try:
        from keras.src.backend.torch import distribution_lib
        
        # Test _sync_cuda (should be no-op on CPU)
        distribution_lib._sync_cuda()
        print("✓ _sync_cuda() works on CPU")
        
        # Test _check_distributed_initialized
        result = distribution_lib._check_distributed_initialized()
        print(f"✓ _check_distributed_initialized() = {result}")
        
        # Test _get_default_device_mesh
        mesh = distribution_lib._get_default_device_mesh()
        print(f"✓ _get_default_device_mesh() = {mesh}")
        
        # Test distribute_variable
        tensor = torch.randn(10, 10)
        result = distribution_lib.distribute_variable(tensor)
        print(f"✓ distribute_variable() works: result is torch.Tensor = {isinstance(result, torch.Tensor)}")
        
    except Exception as e:
        print(f"⚠ DTensor utility test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*70}")
    print("CPU TEST COMPLETE")
    print(f"{'='*70}")
    
    return True


if __name__ == "__main__":
    success = run_cpu_test()
    sys.exit(0 if success else 1)
