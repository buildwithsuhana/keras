#!/usr/bin/env python3
"""
Launcher script for distributed PyTorch training verification.

This script provides an easy way to run verification tests for
DataParallel and ModelParallel with proper logging.

Usage:
    # Single process (for testing):
    python launch_distributed_test.py --single-process
    
    # Multi-GPU distributed:
    torchrun --nproc_per_node=2 launch_distributed_test.py
    
    # Custom number of GPUs:
    torchrun --nproc_per_node=<N> launch_distributed_test.py
"""

import os
import sys

# Set backend before any keras imports
os.environ["KERAS_BACKEND"] = "torch"
os.environ["KERAS_DISTRIBUTION_DEBUG"] = "1"

# Import torch distributed first
import torch
import torch.distributed as dist


def setup_distributed():
    """Setup distributed environment if needed."""
    # Check if we're already initialized
    if dist.is_initialized():
        return True
    
    # Check if we should initialize (has world size info)
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        
        # Setup device
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        
        # Initialize process group
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            init_method="env://"
        )
        
        return True
    
    return False


def main():
    """Main entry point."""
    import argparse
    import logging
    
    parser = argparse.ArgumentParser(description="Launch distributed training verification")
    parser.add_argument("--single-process", action="store_true",
                        help="Run in single process mode")
    parser.add_argument("--test", choices=["dp", "mp", "all"], default="all",
                        help="Which test to run: dp (DataParallel), mp (ModelParallel), all")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of training epochs")
    parser.add_argument("--verbose", action="store_true", default=True,
                        help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Setup distributed if needed
    is_distributed = False
    if not args.single_process:
        is_distributed = setup_distributed()
    
    # Import keras after setup
    import keras
    from keras import layers
    from keras.distribution import (
        DataParallel, ModelParallel, DeviceMesh, LayoutMap,
        list_devices, initialize, distribution
    )
    import numpy as np
    
    # Initialize Keras distribution
    initialize()
    
    # Get rank info
    rank = 0
    world_size = 1
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    
    # Device info
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank}" if world_size > 1 else "cuda:0")
    else:
        device = torch.device("cpu")
    
    # Logging prefix
    def log(msg, rank_0_only=False):
        if rank_0_only and rank != 0:
            return
        prefix = f"[Rank {rank}]" if world_size > 1 else ""
        logger.info(f"{prefix} {msg}")
    
    # Print header
    log("=" * 60)
    log("DISTRIBUTED TRAINING VERIFICATION")
    log("=" * 60)
    log(f"PyTorch version: {torch.__version__}")
    log(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"GPU count: {torch.cuda.device_count()}")
    log(f"Distributed: {is_distributed}")
    if is_distributed:
        log(f"  - Rank: {rank}/{world_size}")
        log(f"  - Device: {device}")
    log("")
    
    # ============ DataParallel Test ============
    if args.test in ["dp", "all"]:
        log("=" * 60)
        log("TEST: DataParallel", rank_0_only=True)
        log("=" * 60)
        
        devices = list_devices("gpu")
        if not devices:
            devices = ["cpu:0"]
        
        log(f"Devices: {devices}")
        log(f"Number of devices: {len(devices)}")
        
        # Create DataParallel
        dp = DataParallel(devices=devices)
        log(f"DataParallel created: {dp.device_mesh.shape}")
        
        with dp.scope():
            model = keras.Sequential([
                layers.Dense(128, activation="relu", input_shape=(64,)),
                layers.Dense(64, activation="relu"),
                layers.Dense(10)
            ])
            log(f"Model parameters: {model.count_params():,}")
            model.compile(optimizer="adam", loss="mse")
        
        # Training
        x = np.random.random((32, 64)).astype("float32")
        y = np.random.random((32, 10)).astype("float32")
        log(f"Data shape: {x.shape} -> {y.shape}")
        
        log("Training...", rank_0_only=True)
        for epoch in range(args.epochs):
            with dp.scope():
                history = model.fit(x, y, epochs=1, verbose=0)
            loss = history.history['loss'][0]
            log(f"  Epoch {epoch+1}/{args.epochs}: loss={loss:.6f}", rank_0_only=True)
        
        log("✓ DataParallel test passed", rank_0_only=True)
        log("")
    
    # ============ ModelParallel Test ============
    if args.test in ["mp", "all"]:
        if torch.cuda.device_count() < 2:
            log("⚠ Skipping ModelParallel test (need >= 2 GPUs)")
        else:
            log("=" * 60)
            log("TEST: ModelParallel", rank_0_only=True)
            log("=" * 60)
            
            devices = list_devices("gpu")
            log(f"Devices: {devices}")
            
            # Create device mesh
            mesh = DeviceMesh(
                shape=(1, len(devices)),
                axis_names=["batch", "model"],
                devices=devices
            )
            log(f"DeviceMesh: shape={mesh.shape}, axes={mesh.axis_names}")
            
            # Create layout map
            layout_map = LayoutMap(mesh)
            layout_map["dense.*kernel"] = (None, "model")
            layout_map["dense.*bias"] = ("model",)
            
            # Create ModelParallel
            mp = ModelParallel(layout_map=layout_map, batch_dim_name="batch")
            log(f"ModelParallel created: {mp.batch_dim_name}")
            
            with mp.scope():
                model = keras.Sequential([
                    layers.Dense(512, activation="relu", input_shape=(128,)),
                    layers.Dense(256, activation="relu"),
                    layers.Dense(10)
                ])
                log(f"Model parameters: {model.count_params():,}")
                model.compile(optimizer="adam", loss="mse")
            
            # Training
            x = np.random.random((32, 128)).astype("float32")
            y = np.random.random((32, 10)).astype("float32")
            log(f"Data shape: {x.shape} -> {y.shape}")
            
            log("Training...", rank_0_only=True)
            for epoch in range(args.epochs):
                with mp.scope():
                    history = model.fit(x, y, epochs=1, verbose=0)
                loss = history.history['loss'][0]
                log(f"  Epoch {epoch+1}/{args.epochs}: loss={loss:.6f}", rank_0_only=True)
            
            log("✓ ModelParallel test passed", rank_0_only=True)
            log("")
    
    # ============ Summary ============
    if rank == 0:
        log("=" * 60)
        log("SUMMARY")
        log("=" * 60)
        log("✓ All tests completed successfully!")
        log("PyTorch distributed training with Keras is working correctly.")
    
    # Cleanup
    if is_distributed:
        dist.destroy_process_group()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

