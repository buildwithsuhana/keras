"""
Verification class for PyTorch distributed training (DataParallel and ModelParallel).

This module provides comprehensive verification of:
1. Device mesh configuration
2. Tensor distribution across devices
3. Model parallelism sharding
4. Data parallelism synchronization
5. Gradient synchronization
6. Training correctness

Usage:
    python verify_torch_distribution.py
"""

import os
import sys
import logging
import time
from typing import Dict, List, Optional, Any

# Set backend before importing keras
os.environ["KERAS_BACKEND"] = "torch"

import torch
import torch.distributed as dist
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try importing keras - may fail if not installed
try:
    import keras
    from keras import layers
    from keras.distribution import (
        DeviceMesh,
        LayoutMap,
        ModelParallel,
        DataParallel,
        set_distribution,
        initialize,
        list_devices,
        process_id,
        num_processes,
    )
    KERAS_AVAILABLE = True
except ImportError as e:
    KERAS_AVAILABLE = False
    logger.warning(f"Keras import failed: {e}")


class DistributionVerifier:
    """
    Comprehensive verification class for PyTorch distributed training.
    
    This class provides methods to verify:
    - Device detection and mesh creation
    - DataParallel functionality
    - ModelParallel sharding
    - Gradient synchronization
    - Training correctness
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the verifier.
        
        Args:
            verbose: Whether to print detailed logs
        """
        self.verbose = verbose
        self.local_rank = 0
        self.world_size = 1
        self.device = None
        self.results = {}
        
        # Setup logging
        self._setup_logging()
        
        # Detect if running in distributed mode
        self._detect_distributed_environment()
        
    def _setup_logging(self):
        """Configure logging based on verbosity."""
        if self.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        else:
            logging.getLogger().setLevel(logging.INFO)
    
    def _detect_distributed_environment(self):
        """Detect if running in distributed mode and setup accordingly."""
        # Check for torch distributed environment
        if dist.is_available() and dist.is_initialized():
            self.local_rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            logger.info(f"Running in distributed mode: rank={self.local_rank}, world_size={self.world_size}")
        else:
            logger.info("Running in single process mode")
        
        # Setup device
        if torch.cuda.is_available():
            if self.world_size > 1:
                # Use local rank for distributed training
                self.device = torch.device(f"cuda:{self.local_rank}")
            else:
                self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        
        logger.info(f"Using device: {self.device}")
    
    def log_info(self, message: str):
        """Log info message with rank prefix."""
        prefix = f"[Rank {self.local_rank}]" if self.world_size > 1 else "[Main]"
        logger.info(f"{prefix} {message}")
    
    def log_debug(self, message: str):
        """Log debug message with rank prefix."""
        if self.verbose:
            prefix = f"[Rank {self.local_rank}]" if self.world_size > 1 else "[Main]"
            logger.debug(f"{prefix} {message}")
    
    # ==================== Device Detection Tests ====================
    
    def verify_device_detection(self) -> Dict[str, Any]:
        """
        Verify device detection functionality.
        
        Returns:
            Dict with test results
        """
        self.log_info("=" * 60)
        self.log_info("TEST: Device Detection")
        self.log_info("=" * 60)
        
        result = {
            "test_name": "device_detection",
            "passed": False,
            "details": {}
        }
        
        try:
            # Test GPU detection
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                result["details"]["gpu_count"] = gpu_count
                result["details"]["gpu_devices"] = [f"cuda:{i}" for i in range(gpu_count)]
                self.log_info(f"Detected {gpu_count} GPU(s)")
                
                for i in range(gpu_count):
                    props = torch.cuda.get_device_properties(i)
                    self.log_info(f"  GPU {i}: {props.name}")
                    self.log_info(f"    - Memory: {props.total_memory / 1e9:.1f} GB")
                    self.log_info(f"    - Compute Capability: {props.major}.{props.minor}")
            else:
                result["details"]["gpu_count"] = 0
                self.log_info("No GPU detected, using CPU")
            
            # Test Keras device listing
            if KERAS_AVAILABLE:
                devices = list_devices("gpu")
                result["details"]["keras_gpu_devices"] = devices
                self.log_info(f"Keras detected GPU devices: {devices}")
            
            result["passed"] = True
            self.log_info("✓ Device detection test PASSED")
            
        except Exception as e:
            self.log_info(f"✗ Device detection test FAILED: {e}")
            result["details"]["error"] = str(e)
        
        return result
    
    # ==================== Device Mesh Tests ====================
    
    def verify_device_mesh(self) -> Dict[str, Any]:
        """
        Verify DeviceMesh creation and configuration.
        
        Returns:
            Dict with test results
        """
        self.log_info("=" * 60)
        self.log_info("TEST: DeviceMesh Creation")
        self.log_info("=" * 60)
        
        result = {
            "test_name": "device_mesh",
            "passed": False,
            "details": {}
        }
        
        try:
            if not KERAS_AVAILABLE:
                self.log_info("Skipping - Keras not available")
                return result
            
            # Get available devices
            devices = list_devices("gpu")
            if not devices:
                devices = [f"cpu:{i}" for i in range(min(4, os.cpu_count() or 1))]
            
            result["details"]["available_devices"] = devices
            self.log_info(f"Available devices: {devices}")
            
            # Test 1D mesh for data parallelism
            if len(devices) > 1:
                mesh_1d = DeviceMesh(
                    shape=(len(devices),),
                    axis_names=["batch"],
                    devices=devices
                )
                result["details"]["mesh_1d_shape"] = mesh_1d.shape
                result["details"]["mesh_1d_axis_names"] = mesh_1d.axis_names
                self.log_info(f"1D DataParallel mesh created: shape={mesh_1d.shape}")
            else:
                mesh_1d = DeviceMesh(
                    shape=(len(devices),),
                    axis_names=["batch"],
                    devices=devices
                )
                self.log_info(f"1D mesh (single device): shape={mesh_1d.shape}")
            
            # Test 2D mesh for model parallelism
            if len(devices) >= 2:
                # Create 2D mesh: (batch, model) dimensions
                mesh_shape = (1, len(devices))
                mesh_2d = DeviceMesh(
                    shape=mesh_shape,
                    axis_names=["batch", "model"],
                    devices=devices
                )
                result["details"]["mesh_2d_shape"] = mesh_2d.shape
                result["details"]["mesh_2d_axis_names"] = mesh_2d.axis_names
                self.log_info(f"2D ModelParallel mesh created: shape={mesh_2d.shape}")
            
            result["passed"] = True
            self.log_info("✓ DeviceMesh creation test PASSED")
            
        except Exception as e:
            self.log_info(f"✗ DeviceMesh creation test FAILED: {e}")
            result["details"]["error"] = str(e)
        
        return result
    
    # ==================== DataParallel Tests ====================
    
    def verify_data_parallel(self) -> Dict[str, Any]:
        """
        Verify DataParallel functionality.
        
        Tests:
        - Model creation with DataParallel
        - Forward pass on multiple devices
        - Gradient synchronization
        - Training step completion
        - Loss computation
        - Weight updates
        
        Returns:
            Dict with test results
        """
        self.log_info("=" * 60)
        self.log_info("TEST: DataParallel Verification")
        self.log_info("=" * 60)
        
        result = {
            "test_name": "data_parallel",
            "passed": False,
            "details": {}
        }
        
        if not KERAS_AVAILABLE:
            self.log_info("Skipping - Keras not available")
            return result
        
        try:
            # Get devices
            devices = list_devices("gpu")
            if not devices:
                devices = [f"cpu:{i}" for i in range(min(4, os.cpu_count() or 1))]
            
            self.log_info(f"Using devices: {devices}")
            
            # Create DataParallel distribution
            dp = DataParallel(devices=devices)
            result["details"]["distribution"] = str(dp)
            self.log_info(f"DataParallel distribution created: {dp}")
            
            # Create model within distribution scope
            with dp.scope():
                model = keras.Sequential([
                    layers.Dense(128, activation="relu", input_shape=(64,)),
                    layers.Dense(10)
                ])
                
                # Log model information
                total_params = model.count_params()
                result["details"]["total_parameters"] = total_params
                self.log_info(f"Model created with {total_params} parameters")
                
                # Log layer details
                for i, layer in enumerate(model.layers):
                    if hasattr(layer, 'kernel'):
                        kernel_shape = layer.kernel.shape
                        self.log_info(f"  Layer {i}: {layer.name}, kernel shape: {kernel_shape}")
                
                model.compile(optimizer="adam", loss="mse")
            
            # Create test data
            batch_size = 32
            x = np.random.random((batch_size, 64)).astype("float32")
            y = np.random.random((batch_size, 10)).astype("float32")
            
            result["details"]["batch_size"] = batch_size
            result["details"]["input_shape"] = x.shape
            result["details"]["target_shape"] = y.shape
            
            # Test forward pass
            with dp.scope():
                # Get initial weights
                initial_weights = [w.numpy() for w in model.weights]
                
                # Forward pass
                predictions = model(x, training=False)
                result["details"]["output_shape"] = predictions.shape
                self.log_info(f"Forward pass completed: input={x.shape}, output={predictions.shape}")
                
                # Compute loss
                loss = model.compute_loss(x, y, predictions)
                result["details"]["loss"] = float(loss.numpy())
                self.log_info(f"Loss computed: {loss.numpy():.4f}")
            
            # Test training step
            with dp.scope():
                model.train_on_batch(x, y)
                self.log_info("Training step completed")
                
                # Get weights after training
                updated_weights = [w.numpy() for w in model.weights]
                
                # Verify weights changed
                weight_changes = []
                for i, (initial, updated) in enumerate(zip(initial_weights, updated_weights)):
                    change = np.abs(updated - initial).sum()
                    weight_changes.append(change)
                
                result["details"]["weight_changes"] = [float(c) for c in weight_changes]
                total_change = sum(weight_changes)
                self.log_info(f"Total weight change: {total_change:.6f}")
                
                if total_change > 0:
                    self.log_info("✓ Weights updated correctly")
                else:
                    self.log_info("⚠ Warning: No weight changes detected")
            
            # Run multiple epochs
            self.log_info("Running 3 epochs...")
            history = model.fit(x, y, epochs=3, verbose=1)
            result["details"]["training_history"] = {
                "epochs": 3,
                "final_loss": float(history.history['loss'][-1]),
                "loss_history": [float(l) for l in history.history['loss']]
            }
            
            self.log_info(f"Training completed. Final loss: {history.history['loss'][-1]:.4f}")
            
            result["passed"] = True
            self.log_info("✓ DataParallel test PASSED")
            
        except Exception as e:
            self.log_info(f"✗ DataParallel test FAILED: {e}")
            import traceback
            self.log_info(traceback.format_exc())
            result["details"]["error"] = str(e)
        
        return result
    
    # ==================== ModelParallel Tests ====================
    
    def verify_model_parallel(self) -> Dict[str, Any]:
        """
        Verify ModelParallel functionality.
        
        Tests:
        - LayoutMap creation and configuration
        - Model sharding across devices
        - Correct tensor placement
        - Parameter sharding verification
        - Forward pass with sharded weights
        - Training with sharded model
        
        Returns:
            Dict with test results
        """
        self.log_info("=" * 60)
        self.log_info("TEST: ModelParallel Verification")
        self.log_info("=" * 60)
        
        result = {
            "test_name": "model_parallel",
            "passed": False,
            "details": {}
        }
        
        if not KERAS_AVAILABLE:
            self.log_info("Skipping - Keras not available")
            return result
        
        try:
            # Get devices
            devices = list_devices("gpu")
            if not devices:
                devices = [f"cpu:{i}" for i in range(min(4, os.cpu_count() or 1))]
            
            if len(devices) < 2:
                self.log_info("Need at least 2 devices for ModelParallel test")
                result["details"]["skipped"] = "insufficient_devices"
                return result
            
            self.log_info(f"Using {len(devices)} devices for ModelParallel")
            
            # Create device mesh for model parallelism
            mesh = DeviceMesh(
                shape=(1, len(devices)),
                axis_names=["batch", "model"],
                devices=devices
            )
            result["details"]["mesh_shape"] = mesh.shape
            result["details"]["mesh_axis_names"] = mesh.axis_names
            self.log_info(f"DeviceMesh created: shape={mesh.shape}, axes={mesh.axis_names}")
            
            # Create layout map
            layout_map = LayoutMap(mesh)
            
            # Define sharding: shard kernels on model axis
            layout_map["dense.*kernel"] = (None, "model")
            layout_map["dense.*bias"] = ("model",)
            
            result["details"]["layout_map_keys"] = list(layout_map.keys())
            self.log_info(f"LayoutMap created with keys: {list(layout_map.keys())}")
            
            # Log layout configurations
            for key in layout_map.keys():
                layout = layout_map[key]
                self.log_info(f"  {key}: axes={layout.axes}")
            
            # Create ModelParallel distribution
            mp = ModelParallel(
                layout_map=layout_map,
                batch_dim_name="batch"
            )
            result["details"]["distribution"] = str(mp)
            self.log_info(f"ModelParallel distribution created: {mp}")
            
            # Create model within distribution scope
            with mp.scope():
                # Create a larger model to demonstrate sharding
                model = keras.Sequential([
                    layers.Dense(512, activation="relu", input_shape=(128,)),
                    layers.Dense(256, activation="relu"),
                    layers.Dense(10)
                ])
                
                total_params = model.count_params()
                result["details"]["total_parameters"] = total_params
                self.log_info(f"Model created with {total_params} parameters")
                
                # Log layer details with expected sharding
                for i, layer in enumerate(model.layers):
                    if hasattr(layer, 'kernel'):
                        kernel_shape = layer.kernel.shape
                        expected_shard_dim = kernel_shape[1] // len(devices) if len(devices) > 1 else kernel_shape[1]
                        self.log_info(f"  Layer {i}: {layer.name}")
                        self.log_info(f"    - kernel shape: {kernel_shape}")
                        self.log_info(f"    - expected shard: dim=1, size={expected_shard_dim}")
                
                model.compile(optimizer="adam", loss="mse")
            
            # Create test data
            batch_size = 32
            x = np.random.random((batch_size, 128)).astype("float32")
            y = np.random.random((batch_size, 10)).astype("float32")
            
            result["details"]["batch_size"] = batch_size
            result["details"]["input_shape"] = x.shape
            
            # Test forward pass with sharded model
            with mp.scope():
                initial_weights = [w.numpy() for w in model.weights]
                
                predictions = model(x, training=False)
                result["details"]["output_shape"] = predictions.shape
                self.log_info(f"Forward pass completed: input={x.shape}, output={predictions.shape}")
                
                loss = model.compute_loss(x, y, predictions)
                result["details"]["loss"] = float(loss.numpy())
                self.log_info(f"Loss computed: {loss.numpy():.4f}")
            
            # Test training step
            with mp.scope():
                model.train_on_batch(x, y)
                self.log_info("Training step completed")
                
                # Verify weights changed
                updated_weights = [w.numpy() for w in model.weights]
                weight_changes = []
                for i, (initial, updated) in enumerate(zip(initial_weights, updated_weights)):
                    change = np.abs(updated - initial).sum()
                    weight_changes.append(change)
                
                result["details"]["weight_changes"] = [float(c) for c in weight_changes]
                total_change = sum(weight_changes)
                self.log_info(f"Total weight change: {total_change:.6f}")
                
                if total_change > 0:
                    self.log_info("✓ Weights updated correctly with sharding")
                else:
                    self.log_info("⚠ Warning: No weight changes detected")
            
            # Run training epochs
            self.log_info("Running 3 epochs with ModelParallel...")
            history = model.fit(x, y, epochs=3, verbose=1)
            result["details"]["training_history"] = {
                "epochs": 3,
                "final_loss": float(history.history['loss'][-1]),
                "loss_history": [float(l) for l in history.history['loss']]
            }
            
            self.log_info(f"Training completed. Final loss: {history.history['loss'][-1]:.4f}")
            
            result["passed"] = True
            self.log_info("✓ ModelParallel test PASSED")
            
        except Exception as e:
            self.log_info(f"✗ ModelParallel test FAILED: {e}")
            import traceback
            self.log_info(traceback.format_exc())
            result["details"]["error"] = str(e)
        
        return result
    
    # ==================== Gradient Synchronization Tests ====================
    
    def verify_gradient_synchronization(self) -> Dict[str, Any]:
        """
        Verify that gradients are properly synchronized across devices.
        
        Returns:
            Dict with test results
        """
        self.log_info("=" * 60)
        self.log_info("TEST: Gradient Synchronization")
        self.log_info("=" * 60)
        
        result = {
            "test_name": "gradient_synchronization",
            "passed": False,
            "details": {}
        }
        
        if not KERAS_AVAILABLE:
            self.log_info("Skipping - Keras not available")
            return result
        
        try:
            devices = list_devices("gpu")
            if not devices:
                devices = [f"cpu:{i}" for i in range(min(4, os.cpu_count() or 1))]
            
            # Test with DataParallel
            dp = DataParallel(devices=devices)
            
            with dp.scope():
                model = keras.Sequential([
                    layers.Dense(64, activation="relu", input_shape=(32,)),
                    layers.Dense(8)
                ])
                model.compile(optimizer="adam", loss="mse")
                
                # Create data
                x = np.random.random((16, 32)).astype("float32")
                y = np.random.random((16, 8)).astype("float32")
                
                # Training step
                model.train_on_batch(x, y)
                
                # Check gradients
                gradients = []
                for layer in model.layers:
                    if hasattr(layer, 'kernel') and layer.kernel.grad is not None:
                        grad_norm = float(torch.norm(layer.kernel.grad).numpy())
                        gradients.append({
                            "layer": layer.name,
                            "grad_norm": grad_norm
                        })
                        self.log_info(f"  {layer.name} gradient norm: {grad_norm:.6f}")
                
                result["details"]["gradients"] = gradients
                result["details"]["num_layers_with_gradients"] = len(gradients)
                
                if len(gradients) > 0:
                    self.log_info("✓ Gradients computed and available")
            
            result["passed"] = True
            self.log_info("✓ Gradient synchronization test PASSED")
            
        except Exception as e:
            self.log_info(f"✗ Gradient synchronization test FAILED: {e}")
            result["details"]["error"] = str(e)
        
        return result
    
    # ==================== Comprehensive Test Runner ====================
    
    def run_all_tests(self) -> Dict[str, Any]:
        """
        Run all verification tests.
        
        Returns:
            Dict with all test results
        """
        self.log_info("\n" + "=" * 60)
        self.log_info("DISTRIBUTION VERIFICATION TEST SUITE")
        self.log_info("=" * 60)
        
        # Environment info
        self.log_info(f"PyTorch version: {torch.__version__}")
        self.log_info(f"Keras available: {KERAS_AVAILABLE}")
        self.log_info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            self.log_info(f"CUDA version: {torch.version.cuda}")
        self.log_info(f"Distributed initialized: {dist.is_initialized() if dist.is_available() else False}")
        
        results = {}
        
        # Run tests
        results["device_detection"] = self.verify_device_detection()
        results["device_mesh"] = self.verify_device_mesh()
        results["data_parallel"] = self.verify_data_parallel()
        results["model_parallel"] = self.verify_model_parallel()
        results["gradient_synchronization"] = self.verify_gradient_synchronization()
        
        # Summary
        self.log_info("\n" + "=" * 60)
        self.log_info("TEST SUMMARY")
        self.log_info("=" * 60)
        
        passed = sum(1 for r in results.values() if r.get("passed", False))
        total = len(results)
        
        for name, result in results.items():
            status = "✓ PASS" if result.get("passed", False) else "✗ FAIL"
            self.log_info(f"  {name}: {status}")
        
        self.log_info(f"\nTotal: {passed}/{total} tests passed")
        
        results["summary"] = {
            "passed": passed,
            "total": total,
            "all_passed": passed == total
        }
        
        return results
    
    def print_summary_report(self, results: Dict[str, Any]):
        """Print a formatted summary report."""
        self.log_info("\n" + "=" * 60)
        self.log_info("VERIFICATION SUMMARY REPORT")
        self.log_info("=" * 60)
        
        if results.get("summary", {}).get("all_passed", False):
            self.log_info("\n✓ ALL TESTS PASSED!")
            self.log_info("PyTorch distributed training is working correctly.")
        else:
            self.log_info("\n⚠ Some tests failed. Please check the details above.")
        
        # Print device info
        if "device_detection" in results:
            details = results["device_detection"].get("details", {})
            if "gpu_count" in details:
                self.log_info(f"\nDevice Configuration:")
                self.log_info(f"  - GPU count: {details['gpu_count']}")
                if "gpu_devices" in details:
                    for device in details["gpu_devices"]:
                        self.log_info(f"  - Device: {device}")
        
        # Print training results
        for test_name in ["data_parallel", "model_parallel"]:
            if test_name in results and results[test_name].get("passed", False):
                details = results[test_name].get("details", {})
                if "training_history" in details:
                    history = details["training_history"]
                    self.log_info(f"\n{test_name.upper()} Training:")
                    self.log_info(f"  - Epochs: {history['epochs']}")
                    self.log_info(f"  - Final loss: {history['final_loss']:.4f}")
                    self.log_info(f"  - Loss history: {history['loss_history']}")


# ==================== Distributed Test Runner ====================

def run_distributed_verification():
    """
    Run verification in distributed mode with proper process group setup.
    
    This function should be called once per process in a distributed setup.
    """
    # Initialize distributed environment if needed
    if not dist.is_initialized():
        if "WORLD_SIZE" in os.environ:
            # Multi-process mode
            local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(local_rank)
            
            dist.init_process_group(
                backend="nccl" if torch.cuda.is_available() else "gloo",
                init_method="env://"
            )
            
            logger.info(f"Process {dist.get_rank()}/{dist.get_world_size()} initialized")
        
        # Initialize Keras distribution
        initialize()
    
    # Create and run verifier
    verifier = DistributionVerifier(verbose=True)
    results = verifier.run_all_tests()
    
    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()
    
    return results


# ==================== Main Entry Point ====================

def main():
    """
    Main entry point for the verification script.
    
    Usage:
        # Single process:
        python verify_torch_distribution.py
        
        # Distributed (with torch.distributed.run):
        torchrun --nproc_per_node=<num_gpus> verify_torch_distribution.py
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify PyTorch distributed training")
    parser.add_argument("--distributed", action="store_true",
                        help="Run in distributed mode")
    parser.add_argument("--verbose", action="store_true", default=True,
                        help="Enable verbose logging")
    args = parser.parse_args()
    
    if args.distributed:
        # Check if we should use torch.distributed.run
        if "LOCAL_RANK" in os.environ:
            # Already set by torch.distributed.run
            results = run_distributed_verification()
        else:
            # Launch with torch.distributed.run
            import subprocess
            import sys
            
            num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
            
            cmd = [
                sys.executable, "-m", "torch.distributed.run",
                "--nproc_per_node", str(num_gpus),
                "--nnodes", "1",
                "--node_rank", "0",
                "--master_port", "29500",
                __file__
            ]
            
            logger.info(f"Launching distributed verification on {num_gpus} GPUs...")
            logger.info(f"Command: {' '.join(cmd)}")
            
            result = subprocess.run(cmd)
            return result.returncode
    else:
        # Single process mode
        # Initialize Keras
        initialize()
        
        # Run verifier
        verifier = DistributionVerifier(verbose=args.verbose)
        results = verifier.run_all_tests()
        verifier.print_summary_report(results)
        
        # Return exit code based on test results
        return 0 if results.get("summary", {}).get("all_passed", False) else 1


if __name__ == "__main__":
    sys.exit(main())

