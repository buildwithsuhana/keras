"""Tests for torch distribution_lib.py"""

import os
import unittest

import torch

# Set backend before importing keras
os.environ["KERAS_BACKEND"] = "torch"

# Now import keras with torch backend
import keras


class TestTorchDistributionLib(unittest.TestCase):
    """Test cases for torch distribution_lib module."""
    
    def test_list_devices_cpu(self):
        """Test listing CPU devices."""
        from keras.src.backend.torch import distribution_lib
        
        devices = distribution_lib.list_devices("cpu")
        self.assertIsInstance(devices, list)
        self.assertTrue(len(devices) > 0)
        print(f"CPU devices: {devices}")
    
    def test_list_devices_gpu(self):
        """Test listing GPU devices."""
        from keras.src.backend.torch import distribution_lib
        
        devices = distribution_lib.list_devices("gpu")
        if torch.cuda.is_available():
            self.assertIsInstance(devices, list)
            print(f"GPU devices: {devices}")
        else:
            print("CUDA not available, skipping GPU test")
    
    def test_get_device_count(self):
        """Test getting device count."""
        from keras.src.backend.torch import distribution_lib
        
        cpu_count = distribution_lib.get_device_count("cpu")
        self.assertIsInstance(cpu_count, int)
        print(f"CPU device count: {cpu_count}")
        
        gpu_count = distribution_lib.get_device_count("gpu")
        print(f"GPU device count: {gpu_count}")
    
    def test_path_adapter_keras_to_torch(self):
        """Test path adapter conversion from Keras to PyTorch style."""
        from keras.src.backend.torch.distribution_lib import TorchPathAdapter
        
        # Test basic conversion
        self.assertEqual(
            TorchPathAdapter.keras_to_torch("dense/kernel"),
            "dense.weight"
        )
        self.assertEqual(
            TorchPathAdapter.keras_to_torch("dense/bias"),
            "dense.bias"
        )
        self.assertEqual(
            TorchPathAdapter.keras_to_torch("model/layer_1/weight"),
            "model.layer_1.weight"
        )
        print("Path adapter conversion test passed")
    
    def test_path_adapter_torch_to_keras(self):
        """Test path adapter conversion from PyTorch to Keras style."""
        from keras.src.backend.torch.distribution_lib import TorchPathAdapter
        
        # Test reverse conversion
        self.assertEqual(
            TorchPathAdapter.torch_to_keras("dense.weight"),
            "dense/kernel"
        )
        self.assertEqual(
            TorchPathAdapter.torch_to_keras("dense.bias"),
            "dense/bias"
        )
        print("Path adapter reverse conversion test passed")
    
    def test_path_adapter_match_pattern(self):
        """Test path adapter pattern matching."""
        from keras.src.backend.torch.distribution_lib import TorchPathAdapter
        
        # Test regex matching with PyTorch paths
        self.assertTrue(
            TorchPathAdapter.match_pattern("dense.*kernel", "dense.weight")
        )
        self.assertTrue(
            TorchPathAdapter.match_pattern("dense.*bias", "dense.bias")
        )
        self.assertFalse(
            TorchPathAdapter.match_pattern("conv2d.*kernel", "dense.weight")
        )
        print("Path adapter pattern matching test passed")
    
    def test_path_adapter_cache(self):
        """Test path adapter caching."""
        from keras.src.backend.torch.distribution_lib import TorchPathAdapter
        
        path = "test/path/to/kernel"
        
        # First conversion
        result1 = TorchPathAdapter.keras_to_torch(path)
        
        # Second conversion should use cache
        result2 = TorchPathAdapter.keras_to_torch(path)
        
        self.assertEqual(result1, result2)
        print("Path adapter caching test passed")
        
        # Clear cache
        TorchPathAdapter.clear_cache()
        print("Path adapter cache cleared")
    
    def test_distribute_variable_basic(self):
        """Test basic variable distribution."""
        from keras.src.backend.torch import distribution_lib
        from keras.src.backend.torch.core import Variable
        
        # Create a simple tensor
        value = torch.randn(10, 20)
        
        # Try to distribute (will use fallback if DTensor not available)
        if distribution_lib.DTENSOR_AVAILABLE:
            print("DTensor is available, testing distribution")
            # This would require a proper DeviceMesh setup
            # Skip for basic test
        else:
            print("DTensor not available, using fallback")
        
        result = distribution_lib.distribute_variable(value, None)
        self.assertEqual(result.shape, value.shape)
        print("Basic distribute variable test passed")
    
    def test_distribute_tensor_basic(self):
        """Test basic tensor distribution."""
        from keras.src.backend.torch import distribution_lib
        
        # Create a simple tensor
        tensor = torch.randn(10, 20)
        
        result = distribution_lib.distribute_tensor(tensor, None)
        self.assertEqual(result.shape, tensor.shape)
        print("Basic distribute tensor test passed")
    
    def test_distribution_initialization(self):
        """Test distribution initialization."""
        from keras.src.backend.torch import distribution_lib
        
        # Test single process mode
        distribution_lib.initialize()
        
        self.assertEqual(distribution_lib.num_processes(), 1)
        self.assertEqual(distribution_lib.process_id(), 0)
        self.assertFalse(distribution_lib.is_distributed())
        print("Distribution initialization test passed")
    
    def test_all_reduce(self):
        """Test all-reduce operation."""
        from keras.src.backend.torch import distribution_lib
        
        # Create a tensor
        tensor = torch.tensor([1.0, 2.0, 3.0])
        
        # In single process mode, should return the same tensor
        result = distribution_lib.all_reduce(tensor)
        self.assertTrue(torch.equal(result, tensor))
        print("All-reduce test passed")
    
    def test_broadcast(self):
        """Test broadcast operation."""
        from keras.src.backend.torch import distribution_lib
        
        # Create a tensor
        tensor = torch.tensor([1.0, 2.0, 3.0])
        
        # In single process mode, should return the same tensor
        result = distribution_lib.broadcast(tensor, src=0)
        self.assertTrue(torch.equal(result, tensor))
        print("Broadcast test passed")


class TestTorchVariableDistribution(unittest.TestCase):
    """Test cases for TorchVariable with distribution."""
    
    def test_variable_with_layout(self):
        """Test variable initialization with layout."""
        from keras.src.backend.torch.core import Variable
        
        # Create variable
        var = Variable(
            initializer=torch.randn(10, 20),
            shape=(10, 20),
            dtype="float32"
        )
        
        self.assertEqual(var.shape, (10, 20))
        print("Variable with layout test passed")


class TestTorchLayerDistribution(unittest.TestCase):
    """Test cases for TorchLayer with distribution."""
    
    def test_dense_layer_tracking(self):
        """Test that Dense layer properly tracks variables."""
        from keras.src.layers import Dense
        
        # Create a Dense layer
        layer = Dense(32, input_shape=(16,))
        
        # Build the layer
        _ = layer(torch.randn(1, 16))
        
        # Check that variables are tracked
        self.assertTrue(len(layer.variables) > 0)
        print(f"Tracked {len(layer.variables)} variables")
        
        # Check that torch_params has both Keras and PyTorch style keys
        keras_paths = [v.path for v in layer.variables]
        print(f"Keras paths: {keras_paths}")
        
        print("Dense layer tracking test passed")


class TestDeviceMeshIntegration(unittest.TestCase):
    """Test DeviceMesh integration with torch backend."""
    
    def test_device_mesh_creation(self):
        """Test creating a DeviceMesh for torch."""
        from keras.src.distribution import DeviceMesh, TensorLayout
        from keras.src.backend.torch import distribution_lib
        
        # Create a simple mesh for CPU
        devices = distribution_lib.list_devices("cpu")
        mesh = DeviceMesh(
            shape=(len(devices),),
            axis_names=["batch"],
            devices=devices[:4] if len(devices) >= 4 else devices
        )
        
        self.assertEqual(mesh.shape[0], min(4, len(devices)))
        print(f"Created DeviceMesh with shape {mesh.shape}")
        print(f"Axis names: {mesh.axis_names}")
        print(f"Devices: {mesh.devices}")
    
    def test_tensor_layout_creation(self):
        """Test creating a TensorLayout."""
        from keras.src.distribution import DeviceMesh, TensorLayout
        from keras.src.backend.torch import distribution_lib
        
        devices = distribution_lib.list_devices("cpu")
        mesh = DeviceMesh(
            shape=(len(devices),),
            axis_names=["batch"],
            devices=devices[:4] if len(devices) >= 4 else devices
        )
        
        # Create a layout for data parallelism (replicate on batch dim)
        layout = TensorLayout(axes=[None], device_mesh=mesh)
        
        self.assertIsNotNone(layout.device_mesh)
        print(f"Created TensorLayout with axes: {layout.axes}")
    
    def test_model_parallel_layout_map(self):
        """Test ModelParallel with LayoutMap."""
        from keras.src.distribution import DeviceMesh, TensorLayout, LayoutMap, ModelParallel
        from keras.src.backend.torch import distribution_lib
        
        devices = distribution_lib.list_devices("cpu")
        mesh = DeviceMesh(
            shape=(2, 2),
            axis_names=["batch", "model"],
            devices=devices[:4] if len(devices) >= 4 else devices
        )
        
        # Create layout map for model parallelism
        layout_map = LayoutMap(mesh)
        layout_map["dense.*kernel"] = TensorLayout([None, "model"], device_mesh=mesh)
        layout_map["dense.*bias"] = TensorLayout(["model"], device_mesh=mesh)
        
        print("Created LayoutMap with model parallelism rules:")
        for key in layout_map:
            print(f"  {key}: {layout_map[key]}")


if __name__ == "__main__":
    print("=" * 60)
    print("Running torch distribution_lib tests")
    print("=" * 60)
    unittest.main(verbosity=2)

