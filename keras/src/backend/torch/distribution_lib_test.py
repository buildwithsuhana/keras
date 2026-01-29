"""Test for torch distribution_lib.py."""

import os
from unittest import mock

import numpy as np
import pytest
import torch

from keras.src import backend
from keras.src import testing
from keras.src.distribution import distribution_lib


@pytest.mark.skipif(
    backend.backend() != "torch",
    reason="Tests for PyTorch backend only",
)
class TorchDistributionLibTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        # Set debug mode for testing
        os.environ["KERAS_DISTRIBUTION_DEBUG"] = "1"

    def tearDown(self):
        super().tearDown()
        os.environ.pop("KERAS_DISTRIBUTION_DEBUG", None)

    def test_list_devices_cpu(self):
        """Test listing CPU devices."""
        devices = distribution_lib.list_devices("cpu")
        self.assertIsInstance(devices, list)
        self.assertGreater(len(devices), 0)
        # CPU devices should be in format cpu:{id}
        for device in devices:
            self.assertTrue(device.startswith("cpu:"))

    def test_list_devices_gpu(self):
        """Test listing GPU devices."""
        if torch.cuda.is_available():
            devices = distribution_lib.list_devices("gpu")
            self.assertIsInstance(devices, list)
            # GPU devices should be in format cuda:{id}
            for device in devices:
                self.assertTrue(device.startswith("cuda:"))
        else:
            pytest.skip("CUDA not available")

    def test_list_devices_all(self):
        """Test listing all devices without specifying type."""
        devices = distribution_lib.list_devices()
        self.assertIsInstance(devices, list)
        self.assertGreater(len(devices), 0)

    def test_get_device_count_cpu(self):
        """Test counting CPU devices."""
        count = distribution_lib.get_device_count("cpu")
        self.assertIsInstance(count, int)
        self.assertGreater(count, 0)

    def test_get_device_count_gpu(self):
        """Test counting GPU devices."""
        if torch.cuda.is_available():
            count = distribution_lib.get_device_count("gpu")
            self.assertIsInstance(count, int)
            self.assertGreater(count, 0)
        else:
            pytest.skip("CUDA not available")

    def test_get_device_count_none(self):
        """Test counting devices with None (default)."""
        count = distribution_lib.get_device_count()
        self.assertIsInstance(count, int)

    def test_initialize_single_process(self):
        """Test initialization in single process mode."""
        # Should not raise any error
        distribution_lib.initialize()
        distribution_lib.initialize(
            job_addresses="10.0.0.1:1234",
            num_processes=1,
            process_id=0,
        )

    def test_num_processes(self):
        """Test getting number of processes."""
        count = distribution_lib.num_processes()
        self.assertIsInstance(count, int)
        self.assertEqual(count, 1)  # Single process by default

    def test_process_id(self):
        """Test getting current process ID."""
        pid = distribution_lib.process_id()
        self.assertIsInstance(pid, int)
        self.assertEqual(pid, 0)  # Process 0 by default

    def test_distribute_tensor(self):
        """Test distributing a tensor."""
        tensor = torch.randn(4, 8)
        
        # Test with no layout (should return as-is)
        result = distribution_lib.distribute_tensor(tensor, None)
        self.assertEqual(result.shape, tensor.shape)

        # Test with tuple layout
        layout = (None, "batch")
        result = distribution_lib.distribute_tensor(tensor, layout)
        self.assertEqual(result.shape, tensor.shape)

    def test_distribute_variable(self):
        """Test distributing a variable."""
        value = np.random.randn(4, 8)
        
        # Test with no layout
        param = distribution_lib.distribute_variable(value, None)
        self.assertIsInstance(param, torch.nn.Parameter)
        self.assertEqual(param.shape, torch.Size(value.shape))

    def test_path_conversion(self):
        """Test path conversion utilities."""
        from keras.src.backend.torch.distribution_lib import (
            keras_to_pytorch_path,
            pytorch_to_keras_path,
            convert_path_for_matching,
        )
        
        # Test basic conversion
        self.assertEqual(
            keras_to_pytorch_path("dense/kernel"),
            "dense.weight"
        )
        self.assertEqual(
            keras_to_pytorch_path("conv2d/bias"),
            "conv2d.bias"
        )
        self.assertEqual(
            keras_to_pytorch_path("my_model/layer1/kernel"),
            "my_model.layer1.weight"
        )
        
        # Test reverse conversion
        self.assertEqual(
            pytorch_to_keras_path("dense.weight"),
            "dense/kernel"
        )
        self.assertEqual(
            pytorch_to_keras_path("conv2d.bias"),
            "conv2d/bias"
        )
        
        # Test convert_path_for_matching
        keras_path, pytorch_path = convert_path_for_matching(
            "dense/kernel", source_format="keras"
        )
        self.assertEqual(keras_path, "dense/kernel")
        self.assertEqual(pytorch_path, "dense.weight")


@pytest.mark.skipif(
    backend.backend() != "torch",
    reason="Tests for PyTorch backend only",
)
class TorchLayoutMapPathAdapterTest(testing.TestCase):
    """Test path adapter functionality in LayoutMap for PyTorch."""

    def setUp(self):
        super().setUp()
        self.devices = [f"cpu:{i}" for i in range(4)]
        self.device_mesh = distribution_lib.DeviceMesh(
            (4,), ["batch"], self.devices
        )

    def test_layout_map_with_keras_path(self):
        """Test LayoutMap with Keras-style paths."""
        layout_map = distribution_lib.LayoutMap(self.device_mesh)
        layout_map["dense/kernel"] = distribution_lib.TensorLayout([None, "batch"])
        layout_map["dense/bias"] = distribution_lib.TensorLayout(["batch"])

        # Should match with Keras path
        layout = layout_map["dense/kernel"]
        self.assertIsNotNone(layout)
        self.assertEqual(layout.axes, (None, "batch"))

        # Should also match with PyTorch path
        layout = layout_map["dense.weight"]
        self.assertIsNotNone(layout)
        self.assertEqual(layout.axes, (None, "batch"))

    def test_layout_map_with_regex_keras(self):
        """Test LayoutMap regex matching with Keras-style paths."""
        layout_map = distribution_lib.LayoutMap(self.device_mesh)
        layout_map[".*kernel"] = distribution_lib.TensorLayout([None, "batch"])
        layout_map[".*bias"] = distribution_lib.TensorLayout(["batch"])

        # Should match Keras path
        layout = layout_map["dense/kernel"]
        self.assertIsNotNone(layout)
        self.assertEqual(layout.axes, (None, "batch"))

        # Should also match PyTorch path
        layout = layout_map["dense.weight"]
        self.assertIsNotNone(layout)
        self.assertEqual(layout.axes, (None, "batch"))

    def test_layout_map_with_pytorch_path(self):
        """Test LayoutMap with PyTorch-style paths."""
        layout_map = distribution_lib.LayoutMap(self.device_mesh)
        layout_map["dense.weight"] = distribution_lib.TensorLayout([None, "batch"])
        layout_map["dense.bias"] = distribution_lib.TensorLayout(["batch"])

        # Should match with PyTorch path
        layout = layout_map["dense.weight"]
        self.assertIsNotNone(layout)
        self.assertEqual(layout.axes, (None, "batch"))

        # Should also match with Keras path
        layout = layout_map["dense/kernel"]
        self.assertIsNotNone(layout)
        self.assertEqual(layout.axes, (None, "batch"))


@pytest.mark.skipif(
    backend.backend() != "torch",
    reason="Tests for PyTorch backend only",
)
class TorchDeviceMeshBackendTest(testing.TestCase):
    """Test DeviceMesh backend conversion for PyTorch."""

    def setUp(self):
        super().setUp()
        os.environ["KERAS_DISTRIBUTION_DEBUG"] = "1"

    def tearDown(self):
        super().tearDown()
        os.environ.pop("KERAS_DISTRIBUTION_DEBUG", None)

    def test_device_mesh_backend_conversion(self):
        """Test converting DeviceMesh to backend mesh."""
        devices = [f"cpu:{i}" for i in range(4)]
        mesh = distribution_lib.DeviceMesh((4,), ["batch"], devices)
        
        # Access backend_mesh property to trigger conversion
        backend_mesh = mesh.backend_mesh
        self.assertIsNotNone(backend_mesh)


@pytest.mark.skipif(
    backend.backend() != "torch",
    reason="Tests for PyTorch backend only",
)
class TorchDataParallelTest(testing.TestCase):
    """Test DataParallel distribution with PyTorch."""

    def setUp(self):
        super().setUp()
        self.devices = [f"cpu:{i}" for i in range(4)]
        self.device_mesh = distribution_lib.DeviceMesh(
            (4,), ["batch"], self.devices
        )

    def test_data_parallel_creation(self):
        """Test creating DataParallel distribution."""
        distribution = distribution_lib.DataParallel(
            device_mesh=self.device_mesh
        )
        self.assertIsNotNone(distribution)
        self.assertEqual(distribution.device_mesh, self.device_mesh)

    def test_data_parallel_data_layout(self):
        """Test getting data layout for DataParallel."""
        distribution = distribution_lib.DataParallel(
            device_mesh=self.device_mesh
        )
        data_shape = (8, 16)
        layout = distribution.get_data_layout(data_shape)
        self.assertIsNotNone(layout)
        self.assertEqual(layout.axes, ("batch", None, None))

    def test_data_parallel_variable_layout(self):
        """Test getting variable layout for DataParallel."""
        distribution = distribution_lib.DataParallel(
            device_mesh=self.device_mesh
        )
        variable = backend.Variable(initializer=np.arange(8))
        layout = distribution.get_variable_layout(variable)
        self.assertIsNotNone(layout)
        # Should be replicated (all None)
        self.assertEqual(layout.axes, (None,))


@pytest.mark.skipif(
    backend.backend() != "torch",
    reason="Tests for PyTorch backend only",
)
class TorchModelParallelTest(testing.TestCase):
    """Test ModelParallel distribution with PyTorch."""

    def setUp(self):
        super().setUp()
        self.devices = [f"cpu:{i}" for i in range(4)]
        self.device_mesh = distribution_lib.DeviceMesh(
            (2, 2), ["data", "model"], self.devices
        )

    def test_model_parallel_creation(self):
        """Test creating ModelParallel distribution."""
        layout_map = distribution_lib.LayoutMap(self.device_mesh)
        layout_map[".*kernel"] = distribution_lib.TensorLayout([None, "model"])
        layout_map[".*bias"] = distribution_lib.TensorLayout(["model"])

        distribution = distribution_lib.ModelParallel(
            layout_map=layout_map,
            batch_dim_name="data",
        )
        self.assertIsNotNone(distribution)
        self.assertEqual(distribution.batch_dim_name, "data")

    def test_model_parallel_variable_layout_keras_path(self):
        """Test getting variable layout with Keras path."""
        layout_map = distribution_lib.LayoutMap(self.device_mesh)
        layout_map["dense/kernel"] = distribution_lib.TensorLayout([None, "model"])
        layout_map["dense/bias"] = distribution_lib.TensorLayout(["model"])

        distribution = distribution_lib.ModelParallel(
            layout_map=layout_map,
            batch_dim_name="data",
        )

        # Test with Keras path
        kernel = backend.Variable(
            initializer=np.random.randn(16, 8),
            name="dense/kernel"
        )
        layout = distribution.get_variable_layout(kernel)
        self.assertIsNotNone(layout)
        self.assertEqual(layout.axes, (None, "model"))

        # Test with PyTorch path (should also work due to adapter)
        kernel2 = backend.Variable(
            initializer=np.random.randn(16, 8),
            name="dense.weight"
        )
        layout2 = distribution.get_variable_layout(kernel2)
        self.assertIsNotNone(layout2)
        self.assertEqual(layout2.axes, (None, "model"))

    def test_model_parallel_data_layout(self):
        """Test getting data layout for ModelParallel."""
        layout_map = distribution_lib.LayoutMap(self.device_mesh)
        layout_map[".*kernel"] = distribution_lib.TensorLayout([None, "model"])

        distribution = distribution_lib.ModelParallel(
            layout_map=layout_map,
            batch_dim_name="data",
        )
        data_shape = (8, 16)
        layout = distribution.get_data_layout(data_shape)
        self.assertIsNotNone(layout)
        self.assertEqual(layout.axes, ("data", None, None))

    def test_model_parallel_tensor_layout(self):
        """Test getting tensor layout for ModelParallel."""
        layout_map = distribution_lib.LayoutMap(self.device_mesh)
        layout_map[".*kernel"] = distribution_lib.TensorLayout([None, "model"])
        layout_map["model/output"] = distribution_lib.TensorLayout(["data", None])

        distribution = distribution_lib.ModelParallel(
            layout_map=layout_map,
            batch_dim_name="data",
        )

        # Test with Keras path
        layout = distribution.get_tensor_layout("model/output")
        self.assertIsNotNone(layout)
        self.assertEqual(layout.axes, ("data", None))

        # Test with PyTorch path (should also work due to adapter)
        layout2 = distribution.get_tensor_layout("model.output")
        self.assertIsNotNone(layout2)
        self.assertEqual(layout2.axes, ("data", None))


@pytest.mark.skipif(
    backend.backend() != "torch",
    reason="Tests for PyTorch backend only",
)
class TorchDistributionScopeTest(testing.TestCase):
    """Test distribution scope management."""

    def test_scope_context_manager(self):
        """Test distribution scope as context manager."""
        devices = [f"cpu:{i}" for i in range(4)]
        device_mesh = distribution_lib.DeviceMesh(
            (4,), ["batch"], devices
        )
        distribution = distribution_lib.DataParallel(
            device_mesh=device_mesh
        )

        # Initially no distribution
        self.assertIsNone(distribution_lib.distribution())

        # Within scope, distribution should be set
        with distribution.scope():
            self.assertEqual(distribution_lib.distribution(), distribution)

        # After exiting scope, distribution should be restored
        self.assertIsNone(distribution_lib.distribution())

    def test_nested_scope(self):
        """Test nested distribution scopes."""
        devices = [f"cpu:{i}" for i in range(4)]
        mesh1 = distribution_lib.DeviceMesh((4,), ["batch"], devices)
        mesh2 = distribution_lib.DeviceMesh((2, 2), ["a", "b"], devices)

        dist1 = distribution_lib.DataParallel(device_mesh=mesh1)
        dist2 = distribution_lib.DataParallel(device_mesh=mesh2)

        with dist1.scope():
            self.assertEqual(distribution_lib.distribution(), dist1)
            with dist2.scope():
                self.assertEqual(distribution_lib.distribution(), dist2)
            self.assertEqual(distribution_lib.distribution(), dist1)

        self.assertIsNone(distribution_lib.distribution())

