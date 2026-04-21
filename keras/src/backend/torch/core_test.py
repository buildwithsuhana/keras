from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch.distributed.tensor import DTensor

from keras.src import backend
from keras.src import testing
from keras.src.backend.common import global_state
from keras.src.backend.torch import core


@pytest.mark.skipif(
    backend.backend() != "torch", reason="Requires Torch backend"
)
class CoreTest(testing.TestCase):
    def test_slice_ops(self):
        # test_slice
        inputs = torch.tensor([[1, 2, 3], [4, 5, 6]])
        result = core.slice(inputs, [0, 1], [2, 2])
        self.assertAllClose(result, [[2, 3], [5, 6]])

        # test_slice_update
        updates = torch.tensor([[10, 20], [30, 40]])
        result = core.slice_update(inputs, [0, 1], updates)
        self.assertAllClose(result, [[1, 10, 20], [4, 30, 40]])

    def test_variable_with_mock_distribution(self):
        from keras.src.distribution import TensorLayout

        mock_dist = MagicMock()
        mock_layout = MagicMock(spec=TensorLayout)
        mock_layout.backend_layout = "mock_layout"
        mock_dist.get_variable_layout.return_value = mock_layout

        orig_get_global_attribute = global_state.get_global_attribute

        def mock_get_global_attribute(attr, default=None, **kwargs):
            if attr == "distribution":
                return mock_dist
            return orig_get_global_attribute(attr, default, **kwargs)

        with patch(
            "keras.src.backend.common.global_state.get_global_attribute",
            side_effect=mock_get_global_attribute,
        ):
            with patch(
                "keras.src.backend.torch.distribution_lib.distribute_tensor",
                return_value=torch.randn(
                    2, 2, device=core.get_device(), dtype=torch.float32
                ),
            ):
                v = core.Variable(
                    np.ones((2, 2), dtype="float32"), dtype="float32"
                )
                self.assertEqual(v._layout, "mock_layout")

                # Test line 120: non-TensorLayout
                mock_dist.get_variable_layout.return_value = "other_layout"
                v2 = core.Variable(
                    np.ones((2, 2), dtype="float32"), dtype="float32"
                )
                self.assertEqual(v2._layout, "other_layout")

                # _direct_assign with layout
                v._direct_assign(np.zeros((2, 2), dtype="float32"))
                self.assertIsInstance(v.value, torch.nn.Parameter)

    def test_variable_dtensor_mock(self):
        # lines 151, 176
        mock_dtensor = MagicMock(spec=DTensor)
        real_tensor = torch.ones(
            (2, 2), device=core.get_device(), dtype=torch.float32
        )
        mock_dtensor.to_local.return_value = real_tensor
        mock_dtensor.detach.return_value = mock_dtensor
        mock_dtensor.shape = torch.Size((2, 2))
        mock_dtensor.device = torch.device(core.get_device())
        mock_dtensor.dtype = torch.float32

        with patch(
            "keras.src.backend.torch.core.convert_to_tensor",
            return_value=mock_dtensor,
        ):
            v = core.Variable(mock_dtensor, shape=(2, 2), dtype="float32")
            mock_dtensor.to_local.assert_called()

            # _direct_assign with DTensor
            # Mock the copy_ on self._value
            v._value = MagicMock()
            v._direct_assign(mock_dtensor)
            mock_dtensor.to_local.assert_called()

    def test_variable_initialize_without_layout_with_parameter(self):
        # Lines 142-144
        param = torch.nn.Parameter(torch.ones(2, 2))
        v = core.Variable(param, trainable=False)
        self.assertFalse(v.value.requires_grad)

        v2 = core.Variable(param, trainable=True)
        self.assertTrue(v2.value.requires_grad)

    def test_variable_value_stateless_scope(self):
        # Lines 219-223
        v = core.Variable([1.0, 2.0])
        from keras.src.backend.common.stateless_scope import StatelessScope

        with StatelessScope() as scope:
            scope.add_update((v, torch.tensor([3.0, 4.0])))
            self.assertAllClose(v.value, [3.0, 4.0])

    def test_variable_uninitialized(self):
        # Line 229
        def initializer(shape, dtype=None):
            return torch.ones(shape, dtype=core.to_torch_dtype(dtype))

        v = core.Variable(initializer=initializer, shape=(2, 2))
        # Accessing .value should trigger initialization
        self.assertAllClose(v.value, torch.ones(2, 2))

    def test_variable_trainable_setter(self):
        # Lines 242-244
        v = core.Variable([1.0, 2.0], trainable=True)
        self.assertTrue(v.value.requires_grad)
        v.trainable = False
        self.assertFalse(v.value.requires_grad)
        self.assertFalse(v.trainable)

    def test_variable_eq_exception(self):
        # Lines 247-250
        v = core.Variable([1.0])
        self.assertFalse(v.__eq__(None))

    def test_convert_to_tensor_errors(self):
        # Lines 255, 257
        with self.assertRaisesRegex(
            ValueError, "not supported with torch backend"
        ):
            core.convert_to_tensor([1], sparse=True)
        with self.assertRaisesRegex(
            ValueError, "not supported with torch backend"
        ):
            core.convert_to_tensor([1], ragged=True)

    def test_convert_to_tensor_variable(self):
        # Line 260
        v = core.Variable([1.0, 2.0])
        res = core.convert_to_tensor(v)
        self.assertAllClose(res, [1.0, 2.0])

    def test_convert_to_tensor_device_and_meta(self):
        # Lines 262-267
        # Test meta tensor to device
        meta_x = torch.empty(2, 2, device="meta")
        res = core.convert_to_tensor(meta_x)
        self.assertEqual(str(res.device.type), core.get_device())

    def test_convert_to_tensor_with_model_parallel_mock(self):
        # lines 314-322
        from keras.src.distribution.distribution_lib import ModelParallel

        mock_dist = MagicMock(spec=ModelParallel)
        mock_mesh = MagicMock()
        mock_mesh.axis_names = []
        mock_dist.device_mesh = mock_mesh

        orig_get_global_attribute = global_state.get_global_attribute

        def mock_get_global_attribute(attr, default=None, **kwargs):
            if attr == "distribution":
                return mock_dist
            return orig_get_global_attribute(attr, default, **kwargs)

        with patch(
            "keras.src.backend.common.global_state.get_global_attribute",
            side_effect=mock_get_global_attribute,
        ):
            with patch(
                "keras.src.backend.torch.distribution_lib.distribute_tensor",
                return_value=torch.randn(2, 2),
            ):
                t = torch.ones((2, 2))
                res = core.convert_to_tensor(t)
                self.assertIsInstance(res, torch.Tensor)

    def test_convert_to_numpy_extras(self):
        # Line 342, 346
        import ml_dtypes

        t = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
        res = core.convert_to_numpy(t)
        self.assertEqual(res.dtype, ml_dtypes.bfloat16)

        res_list = core.convert_to_numpy([torch.tensor(1.0), torch.tensor(2.0)])
        self.assertAllClose(res_list, [1.0, 2.0])

    def test_convert_to_numpy_dtensor_mock(self):
        # line 331
        mock_dtensor = MagicMock(spec=DTensor)
        mock_dtensor.full_tensor.return_value = torch.ones((2, 2), device="cpu")
        mock_dtensor.requires_grad = False
        mock_dtensor.device = torch.device("cpu")
        mock_dtensor.dtype = torch.float32

        res = core.convert_to_numpy(mock_dtensor)
        self.assertAllClose(res, np.ones((2, 2)))
        mock_dtensor.full_tensor.assert_called()

    def test_convert_to_tensor_dtensor_dist(self):
        # hits branch 314->322 (False branch)
        mock_dtensor = MagicMock(spec=DTensor)
        mock_dtensor.device = torch.device(core.get_device())
        mock_dtensor.to.return_value = mock_dtensor
        mock_dtensor.is_meta = False
        mock_dist = MagicMock()
        with patch(
            "keras.src.backend.common.global_state.get_global_attribute",
            return_value=mock_dist,
        ):
            with patch(
                "keras.src.backend.torch.core.is_tensor", return_value=True
            ):
                res = core.convert_to_tensor(mock_dtensor)
                self.assertIs(res, mock_dtensor)
