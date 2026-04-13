import ml_dtypes
import numpy as np
import pytest
import torch

from keras.src import backend
from keras.src import testing
from keras.src.backend.torch import core


@pytest.mark.skipif(backend.backend() != "torch", reason="Requires Torch backend")
class CoreTest(testing.TestCase):
    def test_convert_to_tensor(self):
        # Test basic conversion (use float32 to avoid MPS float64 issues)
        a = np.ones((2, 2), dtype="float32")
        t = core.convert_to_tensor(a)
        self.assertIsInstance(t, torch.Tensor)
        self.assertAllClose(t, a)

        # Test torch-specific: uint32 (converted to int64)
        a_uint32 = np.ones((2, 2), dtype=np.uint32)
        t_uint32 = core.convert_to_tensor(a_uint32)
        self.assertEqual(t_uint32.dtype, torch.int64)

        # Test torch-specific: bfloat16
        a_bf16 = np.ones((2, 2)).astype(ml_dtypes.bfloat16)
        t_bf16 = core.convert_to_tensor(a_bf16)
        self.assertEqual(t_bf16.dtype, torch.bfloat16)

        # Test list of tensors
        t_list = [
            torch.ones((2, 2), device=core.get_device()),
            torch.zeros((2, 2), device=core.get_device()),
        ]
        t_stacked = core.convert_to_tensor(t_list)
        self.assertEqual(t_stacked.shape, (2, 2, 2))

    def test_convert_to_numpy(self):
        # Test bfloat16 conversion (requires special handling in torch)
        t_bf16 = torch.ones(
            (2, 2), dtype=torch.bfloat16, device=core.get_device()
        )
        a_bf16 = core.convert_to_numpy(t_bf16)
        self.assertEqual(a_bf16.dtype, ml_dtypes.bfloat16)
        self.assertAllClose(a_bf16.astype("float32"), np.ones((2, 2)))

        # Test tensor with gradients
        t_grad = torch.ones(
            (2, 2), requires_grad=True, device=core.get_device()
        )
        a_grad = core.convert_to_numpy(t_grad)
        self.assertIsInstance(a_grad, np.ndarray)

    def test_variable(self):
        # Test initialization with torch.nn.Parameter
        # Create it on the correct device so it can be reused
        p = torch.nn.Parameter(
            torch.ones((2, 2), device=core.get_device(), dtype=torch.float32)
        )
        v = core.Variable(p)
        self.assertIs(v.value, p)

        # Test initialization with tensor (should wrap in Parameter)
        t = torch.ones((2, 2), device=core.get_device(), dtype=torch.float32)
        v2 = core.Variable(t)
        self.assertIsInstance(v2.value, torch.nn.Parameter)
        self.assertAllClose(v2.value, t)

        # Test _direct_assign
        v3 = core.Variable(np.zeros((2, 2), dtype="float32"))
        v3._direct_assign(np.ones((2, 2), dtype="float32"))
        self.assertAllClose(v3.value, np.ones((2, 2)))

    def test_compute_output_spec(self):
        from keras.src.backend.common.keras_tensor import KerasTensor

        def fn(x):
            return x + 1

        x = KerasTensor((2, 2), dtype="float32")
        output_spec = core.compute_output_spec(fn, x)
        self.assertIsInstance(output_spec, KerasTensor)
        self.assertEqual(output_spec.shape, (2, 2))
