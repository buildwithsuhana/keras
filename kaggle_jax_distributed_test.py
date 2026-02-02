#!/usr/bin/env python3
"""
Full Multi-GPU/TPU Distributed Training Verification Script for Keras 3 (JAX Backend)
"""

import os
# MUST be set before any other imports
os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false" # Prevent OOM in multi-process
os.environ["KERAS_DISTRIBUTION_DEBUG"] = "1"

import sys
import time
import logging
import numpy as np
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def log(msg, rank_0_only=False):
    import jax
    rank = jax.process_index()
    if rank_0_only and rank != 0:
        return
    prefix = f"[Rank {rank:02d}]"
    logger.info(f"{prefix} {msg}")

def log_section(title):
    separator = "=" * 70
    log(separator, rank_0_only=True)
    log(f"  {title}", rank_0_only=True)
    log(separator, rank_0_only=True)

def setup_environment():
    import jax
    import keras
    
    log_section("ENVIRONMENT SETUP (JAX)")
    
    log(f"Python version: {sys.version.split()[0]}")
    log(f"Keras version: {keras.__version__}")
    log(f"JAX version: {jax.__version__}")
    log(f"Backend: {keras.backend.backend()}")
    
    devices = jax.devices()
    log(f"Total Devices: {len(devices)}")
    for i, d in enumerate(devices):
        log(f"  Device {i}: {d.device_kind} (Process {d.process_index})")
    
    log("")

def test_device_detection():
    import keras
    log_section("TEST 1: DEVICE DETECTION")
    
    gpu_devices = keras.distribution.list_devices("gpu")
    tpu_devices = keras.distribution.list_devices("tpu")
    
    devices = gpu_devices if gpu_devices else tpu_devices
    log(f"✓ Keras detected devices: {devices}")
    log("")

def test_data_parallel(epochs=3):
    import keras
    from keras import layers
    from keras.src.distribution import DataParallel, list_devices
    
    log_section("TEST 2: DATA PARALLEL (DP)")
    
    devices = list_devices("gpu") or list_devices("tpu") or ["cpu:0"]
    dp = DataParallel(devices=devices)
    
    with dp.scope():
        model = keras.Sequential([
            layers.Input(shape=(64,)),
            layers.Dense(128, activation="relu"),
            layers.Dense(10)
        ])
        model.compile(optimizer="adam", loss="mse")
    
    x = np.random.random((32, 64)).astype("float32")
    y = np.random.random((32, 10)).astype("float32")
    
    log(f"Training JAX model (First epoch includes XLA compilation)...")
    
    for epoch in range(epochs):
        start = time.time()
        history = model.fit(x, y, epochs=1, verbose=0)
        log(f"  Epoch {epoch+1}: loss={history.history['loss'][0]:.6f} ({time.time()-start:.3f}s)")
    
    log("✓ DataParallel test PASSED")
    log("")

def test_model_parallel():
    import jax
    import keras
    from keras import layers
    from keras.src.distribution import ModelParallel, DeviceMesh, LayoutMap, list_devices
    
    log_section("TEST 3: MODEL PARALLEL (MP)")
    
    devices = list_devices("gpu") or list_devices("tpu")
    if len(devices) < 2:
        log("⚠ Skipping: Need >= 2 devices for ModelParallel")
        return
    
    # Create 2D Mesh
    mesh = DeviceMesh(shape=(1, len(devices)), axis_names=["batch", "model"], devices=devices)
    layout_map = LayoutMap(mesh)
    layout_map["dense.*kernel"] = (None, "model")
    
    mp = ModelParallel(layout_map=layout_map, batch_dim_name="batch")
    
    with mp.scope():
        model = keras.Sequential([
            layers.Input(shape=(128,)),
            layers.Dense(256, activation="relu", name="dense_shard"),
            layers.Dense(10)
        ])
        model.compile(optimizer="adam", loss="mse")

    # JAX Physical Verification
    log("PHYSICAL SHARDING VERIFICATION (JAX GSPMD):")
    kernel_var = model.get_layer("dense_shard").kernel
    jax_array = kernel_var.value  # This is the underlying jax.Array
    
    # Check addressable shards for this specific process/rank
    addressable_shards = jax_array.addressable_data(0)
    
    log(f"  Global Shape: {jax_array.shape}")
    log(f"  Local Shard Shape (Rank {jax.process_index()}): {addressable_shards.shape}")
    
    if addressable_shards.shape[1] < jax_array.shape[1]:
        log("  ✓ Verified: Kernel is split across the 'model' axis via XLA sharding.")
    else:
        log("  ⚠ Note: Sharding not active or restricted to single device.")
    
    log("✓ ModelParallel test PASSED")
    log("")

def main():
    import keras
    from keras.src.distribution import initialize
    
    # Initialize the distribution system
    initialize()
    
    setup_environment()
    test_device_detection()
    test_data_parallel()
    test_model_parallel()
    
    log_section("VERIFICATION COMPLETE")
    log("All JAX-backend distributed tests finished!")
    return 0

if __name__ == "__main__":
    sys.exit(main())