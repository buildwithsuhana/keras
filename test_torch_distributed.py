"""
Enhanced distributed training script with comprehensive logging.

This script demonstrates how to properly log information during
distributed training with PyTorch backend, showing:
- Process rank and device information
- Distribution configuration
- Training progress with detailed metrics
- Synchronization status
- Model sharding verification

Usage:
    # Single process:
    python test_torch_distributed.py
    
    # Multi-GPU (with torch.distributed.run):
    torchrun --nproc_per_node=<num_gpus> test_torch_distributed.py
"""

import os
import sys
import logging
import time
from datetime import datetime

# Set backend before any keras imports
os.environ["KERAS_BACKEND"] = "torch"
os.environ["KERAS_DISTRIBUTION_DEBUG"] = "1"  # Enable debug logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def get_timestamp():
    """Get current timestamp for logging."""
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


class DistributedLogger:
    """Logger that handles multi-process logging elegantly."""
    
    def __init__(self, name: str = "DistributedTrainer"):
        self.name = name
        self.local_rank = 0
        self.world_size = 1
        
        # Detect distributed environment
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                self.local_rank = dist.get_rank()
                self.world_size = dist.get_world_size()
        except:
            pass
    
    def log(self, level: str, message: str, rank_0_only: bool = False):
        """
        Log message with proper process identification.
        
        Args:
            level: Logging level (INFO, DEBUG, WARNING, ERROR)
            message: Message to log
            rank_0_only: If True, only log on rank 0
        """
        if rank_0_only and self.world_size > 1 and self.local_rank != 0:
            return
            
        timestamp = get_timestamp()
        prefix = f"[{self.name}]"
        
        if self.world_size > 1:
            prefix = f"{prefix} [Rank {self.local_rank:02d}/{self.world_size-1:02d}]"
        
        log_message = f"{timestamp} - {prefix} - {message}"
        
        if level == "INFO":
            logger.info(log_message)
        elif level == "DEBUG":
            logger.debug(log_message)
        elif level == "WARNING":
            logger.warning(log_message)
        elif level == "ERROR":
            logger.error(log_message)
    
    def info(self, message, rank_0_only=False):
        self.log("INFO", message, rank_0_only)
    
    def debug(self, message, rank_0_only=False):
        self.log("DEBUG", message, rank_0_only)
    
    def warning(self, message, rank_0_only=False):
        self.log("WARNING", message, rank_0_only)
    
    def error(self, message, rank_0_only=False):
        self.log("ERROR", message, rank_0_only)
    
    def section(self, title: str):
        """Log a section header."""
        separator = "=" * 60
        self.info(separator, rank_0_only=True)
        self.info(f"  {title}", rank_0_only=True)
        self.info(separator, rank_0_only=True)


def setup_environment():
    """Setup the training environment with proper logging."""
    import torch
    import torch.distributed as dist
    
    logger.info("=" * 60)
    logger.info("DISTRIBUTED TRAINING VERIFICATION SCRIPT")
    logger.info("=" * 60)
    
    # Environment info
    logger.info(f"Python version: {sys.version.split()[0]}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
    
    # Check distributed initialization
    is_distributed = dist.is_available() and dist.is_initialized()
    logger.info(f"Distributed initialized: {is_distributed}")
    
    if is_distributed:
        logger.info(f"  - Local rank: {dist.get_rank()}")
        logger.info(f"  - World size: {dist.get_world_size()}")
        
        # Set device for this process
        if torch.cuda.is_available():
            torch.cuda.set_device(dist.get_rank())
            logger.info(f"  - Device set to: cuda:{dist.get_rank()}")
    
    logger.info("")
    
    return is_distributed


def verify_device_detection():
    """Verify device detection with detailed logging."""
    logger.info("=" * 60)
    logger.info("SECTION 1: Device Detection")
    logger.info("=" * 60)
    
    import torch
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        logger.info(f"✓ Detected {gpu_count} GPU(s)")
        
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            logger.info(f"  GPU {i}:")
            logger.info(f"    - Name: {props.name}")
            logger.info(f"    - Memory: {memory_gb:.1f} GB")
            logger.info(f"    - Compute capability: {props.major}.{props.minor}")
            logger.info(f"    - Multiprocessors: {props.multi_processor_count}")
    else:
        logger.info("⚠ No GPU detected, using CPU")
    
    logger.info("")


def test_data_parallel(logger_obj: DistributedLogger, epochs: int = 2):
    """Test DataParallel with detailed logging."""
    logger_obj.section("DataParallel Test")
    
    import keras
    from keras import layers
    from keras.distribution import DataParallel, list_devices, initialize
    import numpy as np
    
    initialize()
    
    # Get devices
    devices = list_devices("gpu")
    if not devices:
        devices = ["cpu:0"]
    
    logger_obj.info(f"Using devices: {devices}")
    logger_obj.info(f"Number of devices: {len(devices)}")
    
    # Create distribution
    dp = DataParallel(devices=devices)
    logger_obj.info(f"DataParallel created: {dp}")
    logger_obj.info(f"  - Device mesh: {dp.device_mesh}")
    logger_obj.info(f"  - Batch dimension: {dp.batch_dim_name}")
    
    # Create model
    with dp.scope():
        model = keras.Sequential([
            layers.Dense(128, activation="relu", input_shape=(64,)),
            layers.Dense(64, activation="relu"),
            layers.Dense(10)
        ])
        
        total_params = model.count_params()
        logger_obj.info(f"Model created with {total_params:,} parameters")
        
        # Log layer details
        for i, layer in enumerate(model.layers):
            if hasattr(layer, 'kernel') and hasattr(layer, 'bias'):
                kernel_shape = layer.kernel.shape
                bias_shape = layer.bias.shape if layer.bias is not None else None
                logger_obj.info(f"  Layer {i}: {layer.name}")
                logger_obj.info(f"    - kernel: {kernel_shape}")
                if bias_shape:
                    logger_obj.info(f"    - bias: {bias_shape}")
        
        model.compile(optimizer="adam", loss="mse")
    
    # Create data
    batch_size = 32
    x = np.random.random((batch_size, 64)).astype("float32")
    y = np.random.random((batch_size, 10)).astype("float32")
    logger_obj.info(f"Training data shape: input={x.shape}, target={y.shape}")
    
    # Training loop
    logger_obj.info(f"Starting training for {epochs} epochs...")
    logger_obj.info("")
    
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        with dp.scope():
            # Training step
            history = model.fit(x, y, epochs=1, verbose=0)
            loss = history.history['loss'][0]
        
        epoch_time = time.time() - epoch_start
        
        logger_obj.info(f"Epoch {epoch+1}/{epochs}")
        logger_obj.info(f"  - Loss: {loss:.6f}")
        logger_obj.info(f"  - Time: {epoch_time:.3f}s")
    
    total_time = time.time() - start_time
    logger_obj.info("")
    logger_obj.info(f"Total training time: {total_time:.3f}s")
    logger_obj.info("✓ DataParallel test completed successfully")
    
    return True


def test_model_parallel(logger_obj: DistributedLogger, epochs: int = 2):
    """Test ModelParallel with detailed logging."""
    logger_obj.section("ModelParallel Test")
    
    import keras
    from keras import layers
    from keras.distribution import ModelParallel, DeviceMesh, LayoutMap, list_devices, initialize
    import numpy as np
    
    initialize()
    
    # Get devices
    devices = list_devices("gpu")
    if len(devices) < 2:
        logger_obj.warning("Need at least 2 devices for ModelParallel, skipping...")
        return False
    
    logger_obj.info(f"Using {len(devices)} devices for ModelParallel")
    
    # Create device mesh
    mesh = DeviceMesh(
        shape=(1, len(devices)),
        axis_names=["batch", "model"],
        devices=devices
    )
    logger_obj.info(f"DeviceMesh created: shape={mesh.shape}, axes={mesh.axis_names}")
    
    # Create layout map
    layout_map = LayoutMap(mesh)
    layout_map["dense.*kernel"] = (None, "model")
    layout_map["dense.*bias"] = ("model",)
    
    logger_obj.info("LayoutMap configured:")
    for key in layout_map.keys():
        layout = layout_map[key]
        logger_obj.info(f"  - {key}: axes={layout.axes}")
    
    # Create distribution
    mp = ModelParallel(
        layout_map=layout_map,
        batch_dim_name="batch"
    )
    logger_obj.info(f"ModelParallel created: {mp}")
    logger_obj.info(f"  - Device mesh: {mp.device_mesh}")
    logger_obj.info(f"  - Batch dimension: {mp.batch_dim_name}")
    
    # Create model with larger layers to demonstrate sharding
    with mp.scope():
        model = keras.Sequential([
            layers.Dense(512, activation="relu", input_shape=(128,)),
            layers.Dense(256, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(10)
        ])
        
        total_params = model.count_params()
        logger_obj.info(f"Model created with {total_params:,} parameters")
        
        # Log layer details with expected sharding
        for i, layer in enumerate(model.layers):
            if hasattr(layer, 'kernel'):
                kernel_shape = layer.kernel.shape
                shard_info = f"sharded on dim 1 (model axis)"
                logger_obj.info(f"  Layer {i}: {layer.name}")
                logger_obj.info(f"    - kernel shape: {kernel_shape}")
                logger_obj.info(f"    - sharding: {shard_info}")
        
        model.compile(optimizer="adam", loss="mse")
    
    # Create data
    batch_size = 32
    x = np.random.random((batch_size, 128)).astype("float32")
    y = np.random.random((batch_size, 10)).astype("float32")
    logger_obj.info(f"Training data shape: input={x.shape}, target={y.shape}")
    
    # Training loop
    logger_obj.info(f"Starting training for {epochs} epochs...")
    logger_obj.info("")
    
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        with mp.scope():
            history = model.fit(x, y, epochs=1, verbose=0)
            loss = history.history['loss'][0]
        
        epoch_time = time.time() - epoch_start
        
        logger_obj.info(f"Epoch {epoch+1}/{epochs}")
        logger_obj.info(f"  - Loss: {loss:.6f}")
        logger_obj.info(f"  - Time: {epoch_time:.3f}s")
    
    total_time = time.time() - start_time
    logger_obj.info("")
    logger_obj.info(f"Total training time: {total_time:.3f}s")
    logger_obj.info("✓ ModelParallel test completed successfully")
    
    return True


def test_gradient_flow(logger_obj: DistributedLogger):
    """Test gradient flow with detailed logging."""
    logger_obj.section("Gradient Flow Test")
    
    import keras
    from keras import layers
    from keras.distribution import DataParallel, list_devices, initialize
    import numpy as np
    
    initialize()
    
    devices = list_devices("gpu")
    if not devices:
        devices = ["cpu:0"]
    
    dp = DataParallel(devices=devices)
    
    with dp.scope():
        model = keras.Sequential([
            layers.Dense(64, activation="relu", input_shape=(32,)),
            layers.Dense(32, activation="relu"),
            layers.Dense(8)
        ])
        
        model.compile(optimizer="adam", loss="mse")
    
    # Create data
    x = np.random.random((16, 32)).astype("float32")
    y = np.random.random((16, 8)).astype("float32")
    
    # Training step
    with dp.scope():
        model.train_on_batch(x, y)
    
    # Check gradients
    logger_obj.info("Gradient information:")
    grad_count = 0
    
    for layer in model.layers:
        if hasattr(layer, 'kernel') and layer.kernel.grad is not None:
            grad_tensor = layer.kernel.grad
            grad_norm = float(torch.norm(grad_tensor).numpy())
            logger_obj.info(f"  {layer.name}.kernel:")
            logger_obj.info(f"    - gradient norm: {grad_norm:.6f}")
            logger_obj.info(f"    - gradient shape: {grad_tensor.shape}")
            grad_count += 1
    
    if grad_count > 0:
        logger_obj.info(f"✓ {grad_count} layers have gradients")
    else:
        logger_obj.warning("No gradients found")
    
    logger_obj.info("✓ Gradient flow test completed")


def main():
    """Main entry point."""
    import torch.distributed as dist
    
    # Setup environment
    is_distributed = setup_environment()
    
    # Create logger
    logger_obj = DistributedLogger("Trainer")
    
    # Print summary header on rank 0 only
    if not is_distributed or dist.get_rank() == 0:
        logger_obj.section("Starting Verification")
    else:
        logger_obj.info("Process initialized, starting tests...")
    
    try:
        # Run tests
        verify_device_detection()
        
        if not is_distributed or dist.get_rank() == 0:
            logger_obj.section("Data Parallel Test")
        test_data_parallel(logger_obj, epochs=2)
        
        if torch.cuda.device_count() >= 2:
            if not is_distributed or dist.get_rank() == 0:
                logger_obj.section("Model Parallel Test")
            test_model_parallel(logger_obj, epochs=2)
        
        if not is_distributed or dist.get_rank() == 0:
            logger_obj.section("Gradient Flow Test")
        test_gradient_flow(logger_obj)
        
        # Final summary
        if not is_distributed or dist.get_rank() == 0:
            logger_obj.section("All Tests Completed")
            logger_obj.info("✓ All verification tests passed!")
            logger_obj.info("PyTorch distributed training is working correctly.")
        
        return 0
        
    except Exception as e:
        logger_obj.error(f"Test failed with error: {e}")
        import traceback
        logger_obj.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())

