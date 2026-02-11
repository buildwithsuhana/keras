"""Tests for PyTorch backend distribution support.

These tests verify the PyTorch backend implementation of the Keras distribution API,
including path conversion, device mesh creation, and layout mapping.
"""

import os
import pytest

# Set torch backend before importing keras
os.environ["KERAS_BACKEND"] = "torch"

import torch
import numpy as np

import keras
from keras.src.distribution import (
    DeviceMesh,
    TensorLayout,
    LayoutMap,
    ModelParallel,
    DataParallel,
    set_distribution,
    distribution
)
from keras.src.backend.torch import distribution_lib as torch_dist_lib


class TestDeviceDetection:
    """Test device detection and listing."""
    
    def test_list_cpu_devices(self):
        """Test listing CPU devices."""
        devices = torch_dist_lib.list_devices("cpu")
        assert isinstance(devices, list)
        assert len(devices) > 0
        assert all("cpu" in d.lower() for d in devices)
    
    def test_list_gpu_devices(self):
        """Test listing GPU devices."""
        devices = torch_dist_lib.list_devices("gpu")
        if torch.cuda.is_available():
            assert len(devices) == torch.cuda.device_count()
        else:
            # Should still return empty list or cpu fallback
            assert isinstance(devices, list)
    
    def test_get_device_count(self):
        """Test getting device count."""
        cpu_count = torch_dist_lib.get_device_count("cpu")
        assert cpu_count >= 1
        
        gpu_count = torch_dist_lib.get_device_count("gpu")
        assert gpu_count >= 0


class TestPathConversion:
    """Test path conversion between Keras and PyTorch formats."""
    
    def test_keras_to_torch_path_conversion(self):
        """Test conversion from Keras paths to PyTorch paths."""
        _convert = torch_dist_lib._convert_keras_path_to_torch
        
        # Dense layer paths
        assert _convert("dense/kernel") == "dense.weight"
        assert _convert("dense/bias") == "dense.bias"
        
        # Dense layer with index
        assert _convert("dense_1/kernel") == "dense_1.weight"
        assert _convert("dense_1/bias") == "dense_1.bias"
        
        # Conv2D layer
        assert _convert("conv2d/kernel") == "conv2d.weight"
        assert _convert("conv2d/bias") == "conv2d.bias"
        
        # Batch normalization
        assert _convert("batch_normalization/gamma") == "batch_normalization.weight"
        assert _convert("batch_normalization/beta") == "batch_normalization.bias"
        assert _convert("batch_normalization/moving_mean") == "batch_normalization.running_mean"
        assert _convert("batch_normalization/moving_var") == "batch_normalization.running_var"
    
    def test_torch_to_keras_path_conversion(self):
        """Test conversion from PyTorch paths to Keras paths."""
        _convert = torch_dist_lib._convert_torch_path_to_keras
        
        # Dense layer paths
        assert _convert("dense.weight") == "dense/kernel"
        assert _convert("dense.bias") == "dense/bias"
        
        # Batch normalization
        assert _convert("batch_normalization.weight") == "batch_normalization/gamma"
        assert _convert("batch_normalization.bias") == "batch_normalization/beta"
        assert _convert("batch_normalization.running_mean") == "batch_normalization/moving_mean"
        assert _convert("batch_normalization.running_var") == "batch_normalization/moving_variance"


class TestDeviceMesh:
    """Test DeviceMesh creation and manipulation."""
    
    def test_create_device_mesh_cpu(self):
        """Test creating device mesh with CPU devices."""
        devices = ["cpu:0", "cpu:1", "cpu:2", "cpu:3"]
        mesh = DeviceMesh(shape=(4,), axis_names=["batch"], devices=devices)
        
        assert mesh.shape == (4,)
        assert mesh.axis_names == ["batch"]
        assert len(mesh.devices.flatten()) == 4
    
    def test_create_device_mesh_gpu(self):
        """Test creating device mesh with GPU devices."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        devices = [f"cuda:{i}" for i in range(min(4, torch.cuda.device_count()))]
        mesh = DeviceMesh(
            shape=(len(devices),),
            axis_names=["batch"],
            devices=devices
        )
        
        assert mesh.shape == (len(devices),)
        assert len(mesh.devices.flatten()) == len(devices)
    
    def test_create_2d_device_mesh(self):
        """Test creating 2D device mesh for combined parallelism."""
        devices = [f"cuda:{i}" for i in range(4)]
        mesh = DeviceMesh(
            shape=(2, 2),
            axis_names=["batch", "model"],
            devices=devices
        )
        
        assert mesh.shape == (2, 2)
        assert mesh.axis_names == ["batch", "model"]


class TestLayoutMap:
    """Test LayoutMap creation and lookup."""
    
    def test_create_layout_map(self):
        """Test creating a LayoutMap."""
        mesh = DeviceMesh(
            shape=(4,),
            axis_names=["batch"],
            devices=["cpu:0", "cpu:1", "cpu:2", "cpu:3"]
        )
        
        layout_map = LayoutMap(mesh)
        
        # Set some layouts
        layout_map['dense.*kernel'] = (None, 'batch')
        layout_map['dense.*bias'] = ('batch',)
        
        assert len(layout_map) == 2
    
    def test_layout_map_regex_matching(self):
        """Test regex matching in LayoutMap."""
        mesh = DeviceMesh(
            shape=(4,),
            axis_names=["batch"],
            devices=["cpu:0", "cpu:1", "cpu:2", "cpu:3"]
        )
        
        layout_map = LayoutMap(mesh)
        layout_map['dense.*kernel'] = (None, 'batch')
        
        # Test exact match
        assert layout_map['dense/kernel'] is not None
        
        # Test regex match
        assert layout_map['dense_1/kernel'] is not None
        assert layout_map['dense_10/kernel'] is not None
        
        # Test no match
        assert layout_map['conv2d/kernel'] is None
    
    def test_layout_map_tuple_to_tensor_layout(self):
        """Test automatic conversion of tuple to TensorLayout."""
        mesh = DeviceMesh(
            shape=(4,),
            axis_names=["batch"],
            devices=["cpu:0", "cpu:1", "cpu:2", "cpu:3"]
        )
        
        layout_map = LayoutMap(mesh)
        
        # Set using tuple (should be converted to TensorLayout)
        layout_map['dense.*kernel'] = (None, 'batch')
        
        # Check that it's converted
        layout = layout_map['dense/kernel']
        assert isinstance(layout, TensorLayout)
        assert layout.axes == (None, 'batch')


class TestTensorLayout:
    """Test TensorLayout creation and manipulation."""
    
    def test_create_tensor_layout(self):
        """Test creating a TensorLayout."""
        mesh = DeviceMesh(
            shape=(4,),
            axis_names=["batch"],
            devices=["cpu:0", "cpu:1", "cpu:2", "cpu:3"]
        )
        
        # Create layout with explicit axes
        layout = TensorLayout(axes=(None, 'batch'), device_mesh=mesh)
        
        assert layout.axes == (None, 'batch')
        assert layout.device_mesh == mesh
    
    def test_tensor_layout_validation(self):
        """Test TensorLayout axis validation."""
        mesh = DeviceMesh(
            shape=(4,),
            axis_names=["batch"],
            devices=["cpu:0", "cpu:1", "cpu:2", "cpu:3"]
        )
        
        # Valid layout
        layout = TensorLayout(axes=(None, 'batch'), device_mesh=mesh)
        assert layout.axes == (None, 'batch')
        
        # Invalid axis should raise error
        with pytest.raises(ValueError):
            TensorLayout(axes=('invalid',), device_mesh=mesh)


class TestDataParallel:
    """Test DataParallel distribution."""
    
    def test_create_data_parallel(self):
        """Test creating DataParallel distribution."""
        devices = ["cpu:0", "cpu:1", "cpu:2", "cpu:3"]
        
        dp = DataParallel(devices=devices)
        
        assert dp.device_mesh is not None
        assert dp.device_mesh.shape == (4,)
        assert dp.batch_dim_name == "batch"
    
    def test_data_parallel_get_data_layout(self):
        """Test getting data layout from DataParallel."""
        devices = ["cpu:0", "cpu:1", "cpu:2", "cpu:3"]
        
        dp = DataParallel(devices=devices)
        
        # Get data layout for a batch
        layout = dp.get_data_layout(data_shape=(32, 28, 28, 1))
        
        assert layout is not None
        assert isinstance(layout, TensorLayout)
        assert layout.axes[0] == "batch"


class TestModelParallel:
    """Test ModelParallel distribution."""
    
    def test_create_model_parallel(self):
        """Test creating ModelParallel distribution."""
        mesh = DeviceMesh(
            shape=(2, 2),
            axis_names=["batch", "model"],
            devices=["cpu:0", "cpu:1", "cpu:2", "cpu:3"]
        )
        
        layout_map = LayoutMap(mesh)
        layout_map['dense.*kernel'] = (None, 'model')
        
        mp = ModelParallel(layout_map=layout_map)
        
        assert mp.device_mesh == mesh
        assert mp.batch_dim_name == "batch"
    
    def test_model_parallel_variable_layout(self):
        """Test getting variable layout from ModelParallel."""
        mesh = DeviceMesh(
            shape=(2, 2),
            axis_names=["batch", "model"],
            devices=["cpu:0", "cpu:1", "cpu:2", "cpu:3"]
        )
        
        layout_map = LayoutMap(mesh)
        layout_map['dense.*kernel'] = (None, 'model')
        
        mp = ModelParallel(layout_map=layout_map)
        
        # Create a mock variable
        class MockVariable:
            def __init__(self, path):
                self.path = path
                self.shape = (128, 64)
        
        var = MockVariable("dense/kernel")
        layout = mp.get_variable_layout(var)
        
        assert layout is not None
        assert layout.axes == (None, 'model')


class TestDistributionScope:
    """Test distribution context management."""
    
    def test_set_and_get_distribution(self):
        """Test setting and getting distribution."""
        devices = ["cpu:0", "cpu:1"]
        
        dp = DataParallel(devices=devices)
        set_distribution(dp)
        
        # Check that distribution is set
        current = distribution()
        assert current == dp
        
        # Reset
        set_distribution(None)
        current = distribution()
        assert current is None
    
    def test_distribution_scope_context(self):
        """Test distribution scope context manager."""
        devices = ["cpu:0", "cpu:1"]
        
        dp = DataParallel(devices=devices)
        
        # Outside scope
        assert distribution() is None
        
        # Inside scope
        with dp.scope():
            assert distribution() == dp
        
        # Outside scope again
        assert distribution() is None


class TestBackendIntegration:
    """Test integration with PyTorch backend."""
    
    def test_distribution_lib_import(self):
        """Test importing distribution_lib from torch backend."""
        from keras.src.backend.torch import distribution_lib
        
        assert hasattr(distribution_lib, 'list_devices')
        assert hasattr(distribution_lib, 'get_device_count')
        assert hasattr(distribution_lib, 'distribute_variable')
        assert hasattr(distribution_lib, 'distribute_tensor')
    
    def test_device_mesh_backend_conversion(self):
        """Test converting DeviceMesh to backend mesh."""
        mesh = DeviceMesh(
            shape=(4,),
            axis_names=["batch"],
            devices=["cpu:0", "cpu:1", "cpu:2", "cpu:3"]
        )
        
        backend_mesh = torch_dist_lib._to_backend_mesh(mesh)
        
        # Backend mesh should be created
        assert backend_mesh is not None


class TestEndToEndExample:
    """End-to-end tests with actual models."""
    
    def test_simple_model_creation(self):
        """Test creating a simple model with distribution."""
        devices = ["cpu:0", "cpu:1"]
        
        dp = DataParallel(devices=devices)
        set_distribution(dp)
        
        # Create a simple model
        model = keras.Sequential([
            keras.layers.Dense(8, activation='relu', input_shape=(4,)),
            keras.layers.Dense(4)
        ])
        
        # Build model
        model.build(input_shape=(2, 4))
        
        # Check that variables have paths
        for var in model.trainable_variables:
            assert hasattr(var, 'path')
        
        set_distribution(None)
    
    def test_functional_model_creation(self):
        """Test creating a functional model with distribution."""
        devices = ["cpu:0", "cpu:1"]
        
        dp = DataParallel(devices=devices)
        set_distribution(dp)
        
        # Create a functional model
        inputs = keras.Input(shape=(4,))
        x = keras.layers.Dense(8, activation='relu')(inputs)
        outputs = keras.layers.Dense(4)(x)
        model = keras.Model(inputs, outputs)
        
        # Build model
        model.build(input_shape=(2, 4))
        
        # Check variables
        for var in model.trainable_variables:
            assert hasattr(var, 'path')
        
        set_distribution(None)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
