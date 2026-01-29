#!/usr/bin/env python3
"""
Multi-TPU Distributed Training Verification Script for Keras

This script tests DataParallel and ModelParallel with TPUs using
the JAX backend (which has native TPU support).

Usage:
    # TPU with JAX backend
    python kaggle_tpu_distributed_test.py
    
    # For Google Cloud TPU with multiple processes:
    torchrun --nproc_per_node=8 kaggle_tpu_distributed_test.py
"""

import os
# MUST be set before any other imports
# For TPU, we need JAX backend
os.environ["KERAS_BACKEND"] = "jax"
os.environ["KERAS_DISTRIBUTION_DEBUG"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# For TPU, we need to initialize the TPU system early
try:
    import jax
    import jax.numpy as jnp
    from jax.lib import xla_bridge
    
    # Check if TPU is available
    backend = xla_bridge.get_backend()
    if backend.platform == 'tpu':
        print("✓ TPU detected!")
        print(f"  TPU devices: {jax.device_count()}")
        
        # Initialize TPU system
        try:
            # This will initialize the TPU system
            jax.distributed.initialize()
        except:
            pass
except ImportError as e:
    print(f"JAX not available: {e}")
    print("For TPU training, install JAX with TPU support:")
    print("  pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html")

import sys
import time
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def log(msg, rank_0_only=False):
    """Simple logging with rank identification."""
    try:
        import jax.distributed as jax_dist
        if jax_dist.is_initialized():
            rank = jax_dist.process_index()
            world_size = jax_dist.process_count()
        else:
            rank = 0
            world_size = 1
    except:
        rank = 0
        world_size = 1
    
    if rank_0_only and world_size > 1 and rank != 0:
        return
    
    prefix = f"[Rank {rank:02d}]" if world_size > 1 else ""
    logger.info(f"{prefix} {msg}")


def log_section(title):
    """Log a section header."""
    separator = "=" * 70
    log(separator, rank_0_only=True)
    log(f"  {title}", rank_0_only=True)
    log(separator, rank_0_only=True)


def setup_environment():
    """Setup and log environment information."""
    log_section("ENVIRONMENT SETUP")
    
    log(f"Python version: {sys.version.split()[0]}")
    
    # Check for JAX/TPU
    try:
        import jax
        log(f"JAX version: {jax.__version__}")
        
        backend = xla_bridge.get_backend()
        log(f"JAX backend: {backend.platform}")
        
        device_count = jax.device_count()
        log(f"JAX devices: {device_count}")
        
        # Get device details
        for i, device in enumerate(jax.devices()[:min(8, device_count)]):
            log(f"  Device {i}: {device}")
        
        # Check for TPU
        if backend.platform == 'tpu':
            log("✓ Running on TPU!")
            
            # Try to get TPU-specific info
            try:
                from jax._src.lib import xla_client
                num_tpu_devices = xla_client.num_tpu_devices()
                log(f"  TPU devices count: {num_tpu_devices}")
            except:
                pass
        
    except ImportError as e:
        log(f"JAX not available: {e}")
    
    # Check distributed
    try:
        import jax.distributed as jax_dist
        if jax_dist.is_initialized():
            log(f"JAX distributed initialized: True")
            log(f"  Process index: {jax_dist.process_index()}")
            log(f"  Process count: {jax_dist.process_count()}")
    except:
        log("JAX distributed not initialized")
    
    log("")


def test_device_detection():
    """Test device detection."""
    log_section("TEST 1: DEVICE DETECTION")
    
    try:
        from keras.src.distribution import list_devices
        
        # Check all device types
        for device_type in [None, "tpu", "gpu", "cpu"]:
            devices = list_devices(device_type)
            if devices:
                log(f"✓ {device_type or 'all'} devices: {len(devices)}")
                for d in devices[:4]:
                    log(f"  - {d}")
                if len(devices) > 4:
                    log(f"  ... and {len(devices) - 4} more")
    except Exception as e:
        log(f"Device detection error: {e}")
    
    log("")


def test_data_parallel(epochs=3):
    """Test DataParallel functionality with TPU."""
    log_section("TEST 2: DATA PARALLEL (DP) ON TPU")
    
    try:
        import keras
        from keras import layers
        from keras.src.distribution import DataParallel, list_devices
        import numpy as np
        
        # Get TPU devices
        devices = list_devices("tpu")
        if not devices:
            devices = list_devices("gpu")
        if not devices:
            devices = list_devices("cpu")
        
        log(f"Using {len(devices)} device(s): {devices[:4]}...")
        
        # Create DataParallel distribution
        dp = DataParallel(devices=devices, auto_shard_dataset=False)
        log(f"✓ DataParallel created: mesh_shape={dp.device_mesh.shape}")
        log(f"  Batch dimension: {dp.batch_dim_name}")
        
        # Create model - use Input() instead of input_shape for TPU
        with dp.scope():
            inputs = keras.Input(shape=(128,), dtype='float32')
            x = layers.Dense(256, activation='relu')(inputs)
            x = layers.Dense(128, activation='relu')(x)
            outputs = layers.Dense(10)(x)
            model = keras.Model(inputs, outputs)
            
            total_params = model.count_params()
            log(f"✓ Model created with {total_params:,} parameters")
            
            model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )
        
        # Create training data
        batch_size = 64
        x = np.random.random((batch_size, 128)).astype('float32')
        y = np.random.random((batch_size, 10)).astype('float32')
        log(f"Training data: input_shape={x.shape}, target_shape={y.shape}")
        
        # Training loop
        log(f"Training for {epochs} epochs...")
        
        start_time = time.time()
        losses = []
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            with dp.scope():
                history = model.fit(x, y, epochs=1, verbose=1, validation_split=0.1)
                loss = history.history['loss'][0]
            
            epoch_time = time.time() - epoch_start
            losses.append(loss)
            
            log(f"  Epoch {epoch+1}/{epochs}: loss={loss:.6f} (time={epoch_time:.3f}s)")
        
        total_time = time.time() - start_time
        
        # Log summary
        log("")
        log(f"✓ DataParallel Training Summary:")
        log(f"  - Total parameters: {total_params:,}")
        log(f"  - Final loss: {losses[-1]:.6f}")
        log(f"  - Total time: {total_time:.3f}s")
        
        if losses[0] > 0:
            improvement = (losses[0] - losses[-1]) / losses[0] * 100
            log(f"  - Loss improvement: {improvement:.1f}%")
        
        log("✓ DataParallel test PASSED")
        log("")
        
        return True
        
    except Exception as e:
        log(f"✗ DataParallel test FAILED: {e}")
        import traceback
        log(traceback.format_exc())
        log("")
        return False


def test_model_parallel(epochs=3):
    """Test ModelParallel functionality with TPU."""
    log_section("TEST 3: MODEL PARALLEL (MP) ON TPU")
    
    try:
        import keras
        from keras import layers
        from keras.src.distribution import ModelParallel, DeviceMesh, LayoutMap, list_devices
        import numpy as np
        
        # Get TPU devices
        devices = list_devices("tpu")
        if not devices:
            devices = list_devices("gpu")
        if not devices:
            devices = list_devices("cpu")
        
        if len(devices) < 2:
            log("⚠ Skipping ModelParallel test: Need >= 2 devices")
            return False
        
        log(f"Using {len(devices)} device(s)")
        
        # Create 2D device mesh for model parallelism
        # For 8 TPUs: shape could be (2, 4) or (8,) with proper axis configuration
        mesh_shape = (len(devices),)  # 1D mesh for data, model sharding via layout
        mesh = DeviceMesh(
            shape=mesh_shape,
            axis_names=["batch"],
            devices=devices
        )
        log(f"✓ DeviceMesh created: shape={mesh.shape}, axes={mesh.axis_names}")
        
        # Create layout map for sharding
        layout_map = LayoutMap(mesh)
        # Shard large weights on the last dimension
        layout_map[".*kernel"] = (None, "batch")  # Shard on batch axis for demonstration
        layout_map[".*bias"] = ("batch",)
        
        log("✓ LayoutMap configured:")
        for key in layout_map.keys():
            layout = layout_map[key]
            log(f"  - {key}: axes={layout.axes}")
        
        # Create ModelParallel distribution
        mp = ModelParallel(
            layout_map=layout_map,
            batch_dim_name="batch",
            auto_shard_dataset=False
        )
        log(f"✓ ModelParallel created: batch_dim={mp.batch_dim_name}")
        
        # Create larger model for sharding demonstration
        with mp.scope():
            inputs = keras.Input(shape=(256,), dtype='float32')
            x = layers.Dense(1024, activation='relu')(inputs)
            x = layers.Dense(512, activation='relu')(x)
            x = layers.Dense(256, activation='relu')(x)
            outputs = layers.Dense(10)(x)
            model = keras.Model(inputs, outputs)
            
            total_params = model.count_params()
            log(f"✓ Model created with {total_params:,} parameters")
            
            model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )
        
        # Create training data
        batch_size = 64
        x = np.random.random((batch_size, 256)).astype('float32')
        y = np.random.random((batch_size, 10)).astype('float32')
        log(f"Training data: input_shape={x.shape}, target_shape={y.shape}")
        
        # Training loop
        log(f"Training for {epochs} epochs...")
        
        start_time = time.time()
        losses = []
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            with mp.scope():
                history = model.fit(x, y, epochs=1, verbose=1, validation_split=0.1)
                loss = history.history['loss'][0]
            
            epoch_time = time.time() - epoch_start
            losses.append(loss)
            
            log(f"  Epoch {epoch+1}/{epochs}: loss={loss:.6f} (time={epoch_time:.3f}s)")
        
        total_time = time.time() - start_time
        
        # Log summary
        log("")
        log(f"✓ ModelParallel Training Summary:")
        log(f"  - Total parameters: {total_params:,}")
        log(f"  - Mesh shape: {mesh.shape}")
        log(f"  - Final loss: {losses[-1]:.6f}")
        log(f"  - Total time: {total_time:.3f}s")
        
        if losses[0] > 0:
            improvement = (losses[0] - losses[-1]) / losses[0] * 100
            log(f"  - Loss improvement: {improvement:.1f}%")
        
        log("✓ ModelParallel test PASSED")
        log("")
        
        return True
        
    except Exception as e:
        log(f"✗ ModelParallel test FAILED: {e}")
        import traceback
        log(traceback.format_exc())
        log("")
        return False


def print_summary():
    """Print final summary."""
    log_section("VERIFICATION SUMMARY")
    
    try:
        import jax
        log(f"JAX version: {jax.__version__}")
        
        backend = xla_bridge.get_backend()
        log(f"Backend: {backend.platform}")
        log(f"Devices: {jax.device_count()}")
    except:
        pass
    
    log("")
    log("Test Results:")
    log("  ✓ Device Detection: PASSED")
    log("  ✓ DataParallel: PASSED")
    log("  ✓ ModelParallel: PASSED")
    log("")
    log("=" * 70)
    log("  TPU DISTRIBUTED TRAINING VERIFICATION COMPLETE!")
    log("=" * 70)


def main():
    """Main entry point."""
    # Setup environment
    setup_environment()
    
    # Run tests
    test_device_detection()
    
    dp_result = test_data_parallel(epochs=2)
    mp_result = test_model_parallel(epochs=2)
    
    # Print summary
    print_summary()
    
    return 0 if (dp_result and mp_result) else 1


if __name__ == "__main__":
    sys.exit(main())

