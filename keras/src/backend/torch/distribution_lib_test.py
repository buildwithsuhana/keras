"""Tests for PyTorch backend distribution support."""

import pytest

from keras.src import testing
from keras.src.distribution import DeviceMesh
from keras.src.distribution import LayoutMap
from keras.src.distribution import TensorLayout


class TestListDevices(testing.TestCase):
    def test_list_devices_default(self):
        """Test listing all available devices."""
        from keras.src.backend.torch.distribution_lib import list_devices

        devices = list_devices()
        self.assertIsInstance(devices, list)
        # Should have at least CPU devices
        self.assertGreater(len(devices), 0)

    def test_list_devices_gpu(self):
        """Test listing GPU devices."""
        from keras.src.backend.torch.distribution_lib import list_devices

        devices = list_devices("gpu")
        try:
            import torch

            if torch.cuda.is_available():
                self.assertGreater(len(devices), 0)
                for device in devices:
                    self.assertTrue(device.startswith("cuda:"))
            else:
                self.assertEqual(len(devices), 0)
        except ImportError:
            self.skipTest("torch not available")

    def test_list_devices_cpu(self):
        """Test listing CPU devices."""
        from keras.src.backend.torch.distribution_lib import list_devices

        devices = list_devices("cpu")
        self.assertIsInstance(devices, list)
        for device in devices:
            self.assertTrue(device.startswith("cpu:"))


class TestGetDeviceCount(testing.TestCase):
    def test_get_device_count_default(self):
        """Test getting default device count."""
        from keras.src.backend.torch.distribution_lib import get_device_count

        count = get_device_count()
        self.assertIsInstance(count, int)
        self.assertGreaterEqual(count, 1)

    def test_get_device_count_gpu(self):
        """Test getting GPU device count."""
        from keras.src.backend.torch.distribution_lib import get_device_count

        count = get_device_count("gpu")
        self.assertIsInstance(count, int)
        try:
            import torch

            if torch.cuda.is_available():
                self.assertEqual(count, torch.cuda.device_count())
        except ImportError:
            self.skipTest("torch not available")


class TestPathConversion(testing.TestCase):
    def test_convert_keras_path_to_torch(self):
        """Test converting Keras path to PyTorch format."""
        from keras.src.backend.torch.distribution_lib import (
            _convert_keras_path_to_torch,
        )

        self.assertEqual(
            _convert_keras_path_to_torch("dense/kernel"), "dense.weight"
        )
        self.assertEqual(
            _convert_keras_path_to_torch("dense/bias"), "dense.bias"
        )
        self.assertEqual(
            _convert_keras_path_to_torch("conv2d/kernel"),
            "conv2d.weight",
        )

    def test_convert_torch_path_to_keras(self):
        """Test converting PyTorch path to Keras format."""
        from keras.src.backend.torch.distribution_lib import (
            _convert_torch_path_to_keras,
        )

        self.assertEqual(
            _convert_torch_path_to_keras("dense.weight"), "dense/kernel"
        )
        self.assertEqual(
            _convert_torch_path_to_keras("dense.bias"), "dense/bias"
        )


class TestBackendMeshConversion(testing.TestCase):
    def test_to_backend_mesh(self):
        """Test converting DeviceMesh to PyTorch DeviceMesh."""
        pytest.importorskip("torch")
        try:
            from torch.distributed.device_mesh import DeviceMesh
        except ImportError:
            pytest.skip("torch.distributed not available")

        from keras.src.backend.torch.distribution_lib import _to_backend_mesh
        from keras.src.distribution import DeviceMesh as KerasDeviceMesh

        # Create a simple Keras DeviceMesh
        devices = ["cpu:0", "cpu:1"]
        keras_mesh = KerasDeviceMesh(
            shape=(2,), axis_names=["data"], devices=devices
        )

        # Convert to PyTorch mesh
        torch_mesh = _to_backend_mesh(keras_mesh)

        self.assertIsNotNone(torch_mesh)
        self.assertEqual(torch_mesh.shape["data"], 2)


class TestBackendLayoutConversion(testing.TestCase):
    def test_to_backend_layout(self):
        """Test converting TensorLayout to PyTorch placements."""
        pytest.importorskip("torch")
        try:
            from torch.distributed.device_mesh import DeviceMesh
        except ImportError:
            pytest.skip("torch.distributed not available")

        from keras.src.backend.torch.distribution_lib import (
            _to_backend_layout,
        )
        from keras.src.distribution import DeviceMesh as KerasDeviceMesh
        from keras.src.distribution import TensorLayout

        # Create a Keras DeviceMesh
        devices = ["cpu:0", "cpu:1"]
        keras_mesh = KerasDeviceMesh(
            shape=(2,), axis_names=["data"], devices=devices
        )

        # Create a TensorLayout
        layout = TensorLayout(axes=(None, "data"), device_mesh=keras_mesh)

        # Convert to PyTorch placements
        placements = _to_backend_layout(layout)

        self.assertIsInstance(placements, tuple)


class TestModelParallelHelper(testing.TestCase):
    def test_model_parallel_init_with_tuple(self):
        """Test ModelParallel initialization with tuple."""
        pytest.importorskip("torch")
        try:
            from torch.distributed.device_mesh import DeviceMesh
        except ImportError:
            pytest.skip("torch.distributed not available")

        from keras.src.backend.torch.distribution_lib import ModelParallel

        model_parallel = ModelParallel(device_mesh=(2, 2))

        self.assertIsNotNone(model_parallel.mesh)
        self.assertEqual(model_parallel.mesh.shape["data"], 2)
        self.assertEqual(model_parallel.mesh.shape["model"], 2)

    def test_infer_parallel_style_linear(self):
        """Test inferring parallel style for Linear layer."""
        pytest.importorskip("torch")
        try:
            from torch.distributed.device_mesh import DeviceMesh
        except ImportError:
            pytest.skip("torch.distributed not available")

        import torch.nn as nn

        from keras.src.backend.torch.distribution_lib import ModelParallel

        model_parallel = ModelParallel(device_mesh=(2, 2))
        linear = nn.Linear(8, 16)

        # Test ColwiseParallel (shard output dim)
        style = model_parallel._infer_parallel_style(
            linear, "weight", (None, "model")
        )
        # Note: Actual style depends on torch distributed tensor version
        self.assertIsNotNone(style)

    def test_infer_parallel_style_conv2d(self):
        """Test inferring parallel style for Conv2D layer."""
        pytest.importorskip("torch")
        try:
            from torch.distributed.device_mesh import DeviceMesh
        except ImportError:
            pytest.skip("torch.distributed not available")

        import torch.nn as nn

        from keras.src.backend.torch.distribution_lib import ModelParallel

        model_parallel = ModelParallel(device_mesh=(2, 2))
        conv = nn.Conv2d(3, 64, kernel_size=3)

        # Test ColwiseParallel (shard output channels)
        style = model_parallel._infer_parallel_style(
            conv, "weight", (None, None, None, "model")
        )
        self.assertIsNotNone(style)


class TestDistributeTensor(testing.TestCase):
    def test_distribute_tensor_basic(self):
        """Test basic tensor distribution."""
        pytest.importorskip("torch")
        try:
            from torch.distributed.device_mesh import DeviceMesh
        except ImportError:
            pytest.skip("torch.distributed not available")

        import torch

        from keras.src.backend.torch.distribution_lib import (
            distribute_tensor,
        )
        from keras.src.distribution import DeviceMesh as KerasDeviceMesh
        from keras.src.distribution import TensorLayout

        # Create a simple device mesh
        devices = ["cpu:0", "cpu:1"]
        keras_mesh = KerasDeviceMesh(
            shape=(2,), axis_names=["data"], devices=devices
        )

        # Create a layout
        layout = TensorLayout(axes=(None, "data"), device_mesh=keras_mesh)

        # Create a tensor
        tensor = torch.randn(4, 8)

        # Distribute the tensor
        distributed = distribute_tensor(tensor, layout)

        # Should return a tensor (DTensor if available)
        self.assertIsInstance(distributed, torch.Tensor)


class TestLayoutMapIntegration(testing.TestCase):
    def test_layout_map_with_torch_path(self):
        """Test LayoutMap with PyTorch-style paths."""
        from keras.src.backend.torch.distribution_lib import (
            _convert_keras_path_to_torch,
        )

        # Test path conversion works correctly
        torch_path = _convert_keras_path_to_torch("dense/kernel")
        self.assertEqual(torch_path, "dense.weight")


class TestEndToEndModelParallel(testing.TestCase):
    def test_e2e_model_parallel_creation(self):
        """Test end-to-end ModelParallel creation and usage."""
        pytest.importorskip("torch")
        try:
            from torch.distributed.device_mesh import DeviceMesh
        except ImportError:
            pytest.skip("torch.distributed not available")

        import torch
        import torch.nn as nn

        from keras.src.backend.torch.distribution_lib import ModelParallel
        from keras.src.distribution import DeviceMesh as KerasDeviceMesh
        from keras.src.distribution import LayoutMap
        from keras.src.distribution import TensorLayout

        # Create a device mesh
        devices = ["cpu:0", "cpu:1", "cpu:2", "cpu:3"]
        device_mesh = KerasDeviceMesh(
            shape=(2, 2), axis_names=["batch", "model"], devices=devices
        )

        # Create a layout map
        layout_map = LayoutMap(device_mesh)
        layout_map["dense.*kernel"] = TensorLayout([None, "model"])
        layout_map["dense.*bias"] = TensorLayout(["model"])

        # Create ModelParallel helper
        model_parallel = ModelParallel(
            device_mesh=(2, 2), layout_map=layout_map
        )

        self.assertIsNotNone(model_parallel.mesh)
        self.assertEqual(model_parallel.mesh.shape["data"], 2)
        self.assertEqual(model_parallel.mesh.shape["model"], 2)

        # Test with a simple module
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.dense = nn.Linear(8, 16)

            def forward(self, x):
                return self.dense(x)

        model = SimpleModel()

        # Apply parallelization
        parallelized_model = model_parallel.parallelize_module(model)

        self.assertIsNotNone(parallelized_model)


class TestDTensorImports(testing.TestCase):
    def test_dtensor_availability(self):
        """Test that DTensor imports work correctly."""
        try:
            from torch.distributed.tensor import distribute_tensor, DTensor

            self.assertTrue(True)
        except ImportError:
            # DTensor might not be available in older PyTorch versions
            self.skipTest("DTensor not available in this PyTorch version")

    def test_parallel_styles_availability(self):
        """Test that parallel styles import correctly."""
        try:
            from torch.distributed.tensor.parallel import (
                ColwiseParallel,
                RowwiseParallel,
            )

            self.assertTrue(True)
        except ImportError:
            self.skipTest(
                "torch.distributed.tensor.parallel not available"
            )

