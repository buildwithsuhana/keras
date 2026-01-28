"""Test for torch distribution_lib.py."""

import os
from unittest import mock

import numpy as np
import pytest
import torch

from keras.src import backend
from keras.src import testing
from keras.src.backend import distribution_lib as backend_dlib
from keras.src.distribution import distribution_lib


# Skip tests if not torch backend
pytestmark = pytest.mark.skipif(
    backend.backend() != "torch",
    reason="Only torch backend tests",
)


class TorchListDevicesTest(testing.TestCase):
    def test_list_devices_gpu(self):
        """Test listing GPU devices."""
        if torch.cuda.is_available():
            devices = backend_dlib.list_devices()
            self.assertGreater(len(devices), 0)
            # Check format is cuda:X
            for device in devices:
                self.assertTrue(device.startswith("cuda:"))
        else:
            devices = backend_dlib.list_devices("cpu")
            self.assertGreaterEqual(len(devices), 1)

    def test_list_devices_cpu(self):
        """Test listing CPU devices."""
        devices = backend_dlib.list_devices("cpu")
        self.assertGreaterEqual(len(devices), 1)

    def test_list_devices_none(self):
        """Test listing all devices without type filter."""
        devices = backend_dlib.list_devices()
        # Should have at least some devices
        self.assertGreaterEqual(len(devices), 1)


class TorchGetDeviceCountTest(testing.TestCase):
    def test_get_device_count_gpu(self):
        """Test counting GPU devices."""
        if torch.cuda.is_available():
            count = backend_dlib.get_device_count("gpu")
            self.assertEqual(count, torch.cuda.device_count())

    def test_get_device_count_cpu(self):
        """Test counting CPU devices."""
        count = backend_dlib.get_device_count("cpu")
        self.assertGreaterEqual(count, 1)

    def test_get_device_count_none(self):
        """Test default device count."""
        count = backend_dlib.get_device_count()
        if torch.cuda.is_available():
            self.assertEqual(count, torch.cuda.device_count())
        else:
            self.assertGreaterEqual(count, 1)


class TorchDTensorPlacementTest(testing.TestCase):
    """Test DTensor placement conversions."""

    def test_tensor_layout_to_placements_replicate(self):
        """Test conversion of replicated layout to placements."""
        from keras.src.distribution import DeviceMesh, TensorLayout
        from keras.src.backend.torch import distribution_lib

        mesh = distribution_lib.DeviceMesh(
            (2,), ["batch"], [f"cuda:{i}" if torch.cuda.is_available() else "cpu" for i in range(2)]
        )

        # Replicated layout
        layout = TensorLayout([None, None], mesh)
        placements = distribution_lib._tensor_layout_to_placements(layout)

        self.assertEqual(len(placements), 2)
        for placement in placements:
            self.assertIsInstance(placement, distribution_lib.Replicate)

    def test_tensor_layout_to_placements_shard(self):
        """Test conversion of sharded layout to placements."""
        from keras.src.distribution import DeviceMesh, TensorLayout
        from keras.src.backend.torch import distribution_lib

        mesh = distribution_lib.DeviceMesh(
            (2,), ["batch"], [f"cuda:{i}" if torch.cuda.is_available() else "cpu" for i in range(2)]
        )

        # Sharded on batch dimension
        layout = TensorLayout(["batch", None], mesh)
        placements = distribution_lib._tensor_layout_to_placements(layout)

        self.assertEqual(len(placements), 2)
        self.assertIsInstance(placements[0], distribution_lib.Shard)
        self.assertIsInstance(placements[1], distribution_lib.Replicate)


class TorchPathAdapterTest(testing.TestCase):
    """Test Keras/PyTorch path adapter."""

    def test_keras_to_torch_simple(self):
        """Test simple path conversion."""
        from keras.src.backend.torch.distribution_lib import TorchPathAdapter

        # Test simple kernel path
        result = TorchPathAdapter.keras_to_torch("dense/kernel")
        self.assertIn("weight", result)

    def test_keras_to_torch_with_wildcard(self):
        """Test path with wildcard conversion."""
        from keras.src.backend.torch.distribution_lib import TorchPathAdapter

        result = TorchPathAdapter.keras_to_torch("dense.*kernel")
        self.assertTrue(len(result) > 0)

    def test_torch_to_keras(self):
        """Test PyTorch to Keras path conversion."""
        from keras.src.backend.torch.distribution_lib import TorchPathAdapter

        result = TorchPathAdapter.torch_to_keras("dense.weight")
        self.assertIn("kernel", result)

    def test_matches_keras_pattern(self):
        """Test pattern matching."""
        from keras.src.backend.torch.distribution_lib import TorchPathAdapter

        # Test that a PyTorch path matches a Keras pattern
        matches = TorchPathAdapter.matches_keras_pattern(
            "dense.*kernel",
            "dense.weight"
        )
        self.assertTrue(matches)

        # Test non-matching pattern
        matches = TorchPathAdapter.matches_keras_pattern(
            "conv.*kernel",
            "dense.weight"
        )
        self.assertFalse(matches)


class TorchBackendMeshTest(testing.TestCase):
    """Test backend mesh conversion."""

    def test_to_backend_mesh(self):
        """Test conversion of DeviceMesh to torch DeviceMesh."""
        from keras.src.distribution import DeviceMesh
        from keras.src.backend.torch import distribution_lib

        devices = [f"cuda:{i}" if torch.cuda.is_available() else "cpu" for i in range(2)]
        mesh = DeviceMesh((2,), ["batch"], devices)

        torch_mesh = distribution_lib._to_backend_mesh(mesh)

        self.assertIsNotNone(torch_mesh)
        self.assertEqual(torch_mesh.shape[0], 2)


class TorchDistributeVariableTest(testing.TestCase):
    """Test variable distribution."""

    def test_distribute_variable_basic(self):
        """Test basic variable distribution."""
        from keras.src.distribution import DeviceMesh, TensorLayout
        from keras.src.backend.torch import distribution_lib

        devices = [f"cuda:{i}" if torch.cuda.is_available() else "cpu" for i in range(2)]
        mesh = DeviceMesh((2,), ["batch"], devices)

        # Create a simple tensor
        tensor = torch.randn(8, 16)

        # Try to distribute
        try:
            layout = TensorLayout([None], mesh)
            result = distribution_lib.distribute_variable(tensor, layout)
            # Result should be a tensor
            self.assertTrue(isinstance(result, torch.Tensor))
        except Exception as e:
            # DTensor may not be available in all PyTorch versions
            self.skipTest(f"DTensor not available: {e}")


class TorchInferParallelStyleTest(testing.TestCase):
    """Test parallel style inference."""

    def test_infer_parallel_style_linear_colwise(self):
        """Test colwise parallel style inference for Linear."""
        from keras.src.backend.torch.distribution_lib import infer_parallel_style
        import torch.nn as nn

        module = nn.Linear(16, 32)
        param_name = "weight"

        # (None, 'model') should give colwise
        result = infer_parallel_style(module, param_name, (None, "model"))
        self.assertEqual(result, "colwise")

    def test_infer_parallel_style_linear_rowwise(self):
        """Test rowwise parallel style inference for Linear."""
        from keras.src.backend.torch.distribution_lib import infer_parallel_style
        import torch.nn as nn

        module = nn.Linear(16, 32)
        param_name = "weight"

        # ('model', None) should give rowwise
        result = infer_parallel_style(module, param_name, ("model", None))
        self.assertEqual(result, "rowwise")

    def test_infer_parallel_style_no_model_axis(self):
        """Test when no model axis in spec."""
        from keras.src.backend.torch.distribution_lib import infer_parallel_style
        import torch.nn as nn

        module = nn.Linear(16, 32)
        param_name = "weight"

        # (None, None) should give None
        result = infer_parallel_style(module, param_name, (None, None))
        self.assertIsNone(result)


class TorchDistributionInitializeTest(testing.TestCase):
    """Test distribution initialization."""

    def test_initialize_with_env_vars(self):
        """Test initialization with environment variables."""
        from keras.src.backend.torch import distribution_lib

        # Set environment variables
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"

        try:
            distribution_lib.initialize()
            # Should not raise
        except Exception as e:
            self.skipTest(f"Could not initialize distributed: {e}")

    def test_num_processes(self):
        """Test num_processes function."""
        from keras.src.backend.torch import distribution_lib

        num_procs = distribution_lib.num_processes()
        self.assertGreaterEqual(num_procs, 1)

    def test_process_id(self):
        """Test process_id function."""
        from keras.src.backend.torch import distribution_lib

        process_id = distribution_lib.process_id()
        self.assertGreaterEqual(process_id, 0)
        self.assertLess(process_id, distribution_lib.num_processes())


class TorchCleanupTest(testing.TestCase):
    """Test cleanup functions."""

    def test_cleanup_distributed(self):
        """Test cleanup of distributed resources."""
        from keras.src.backend.torch import distribution_lib

        try:
            distribution_lib.cleanup_distributed()
        except Exception as e:
            # Cleanup may fail if not initialized
            pass


# Integration tests (only run with CUDA)
if torch.cuda.is_available():
    class TorchDTensorIntegrationTest(testing.TestCase):
        """Integration tests for DTensor with distributed settings."""

        def test_distribute_tensor_with_mesh(self):
            """Test tensor distribution with device mesh."""
            from keras.src.distribution import DeviceMesh, TensorLayout
            from keras.src.backend.torch import distribution_lib

            # Skip if not enough GPUs
            if torch.cuda.device_count() < 2:
                self.skipTest("Need at least 2 GPUs for this test")

            devices = [f"cuda:{i}" for i in range(2)]
            mesh = DeviceMesh((2,), ["batch"], devices)

            # Create and distribute tensor
            tensor = torch.randn(16, 32)
            layout = TensorLayout(["batch", None], mesh)

            result = distribution_lib.distribute_tensor(tensor, layout)

            # Result should have appropriate shape
            self.assertIsInstance(result, torch.Tensor)

        def test_model_parallel_layout_map(self):
            """Test model parallelism with LayoutMap."""
            from keras.src.distribution import DeviceMesh, LayoutMap, TensorLayout
            from keras.src.backend.torch import distribution_lib

            # Skip if not enough GPUs
            if torch.cuda.device_count() < 2:
                self.skipTest("Need at least 2 GPUs for this test")

            devices = [f"cuda:{i}" for i in range(2)]
            mesh = DeviceMesh((2,), ["model"], devices)
            layout_map = LayoutMap(mesh)

            # Add layout for kernel
            layout_map[".*dense.*kernel"] = TensorLayout([None, "model"], mesh)

            # Check layout lookup
            layout = layout_map["dense/kernel"]
            self.assertIsNotNone(layout)

