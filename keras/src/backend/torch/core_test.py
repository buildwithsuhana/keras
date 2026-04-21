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
        inputs = torch.tensor([[1, 2, 3], [4, 5, 6]])
        self.assertAllClose(
            core.slice(inputs, [0, 1], [2, 2]), [[2, 3], [5, 6]]
        )
        self.assertAllClose(
            core.slice_update(
                inputs, [0, 1], torch.tensor([[10, 20], [30, 40]])
            ),
            [[1, 10, 20], [4, 30, 40]],
        )

    def test_variable_basics(self):
        # Basics: trainable, eq, uninitialized
        v = core.Variable([1.0, 2.0], trainable=True)
        self.assertTrue(v.value.requires_grad)
        v.trainable = False
        self.assertFalse(v.value.requires_grad or v.trainable)
        self.assertFalse(v.__eq__(None))
        v2 = core.Variable(lambda s, **k: torch.ones(s), shape=(2, 2))
        self.assertAllClose(v2.value, torch.ones(2, 2))
        # From Parameter
        param = torch.nn.Parameter(torch.ones(2, 2))
        self.assertFalse(
            core.Variable(param, trainable=False).value.requires_grad
        )
        self.assertTrue(
            core.Variable(param, trainable=True).value.requires_grad
        )
        # Stateless
        from keras.src.backend.common.stateless_scope import StatelessScope

        with StatelessScope() as scope:
            scope.add_update((v, torch.tensor([3.0, 4.0])))
            self.assertAllClose(v.value, [3.0, 4.0])

    def test_variable_distribution_mocking(self):
        from keras.src.distribution import TensorLayout

        mock_dist = MagicMock()
        mock_layout = MagicMock(spec=TensorLayout, backend_layout="mock_layout")
        mock_dist.get_variable_layout.return_value = mock_layout
        orig_get = global_state.get_global_attribute

        def side_effect(attr, *a, **k):
            if attr == "distribution":
                return mock_dist
            return orig_get(attr, *a, **k)

        with (
            patch(
                "keras.src.backend.common.global_state.get_global_attribute",
                side_effect=side_effect,
            ),
            patch(
                "keras.src.backend.torch.distribution_lib.distribute_tensor",
                return_value=torch.randn(2, 2),
            ),
        ):
            v = core.Variable(np.ones((2, 2), "float32"), dtype="float32")
            self.assertEqual(v._layout, "mock_layout")
            mock_dist.get_variable_layout.return_value = "other_layout"
            self.assertEqual(
                core.Variable(
                    np.ones((2, 2), "float32"), dtype="float32"
                )._layout,
                "other_layout",
            )
            v._direct_assign(np.zeros((2, 2), "float32"))
            self.assertIsInstance(v.value, torch.nn.Parameter)

    def test_variable_dtensor_mock(self):
        mock_dt = MagicMock(
            spec=DTensor, shape=torch.Size((2, 2)), dtype=torch.float32
        )
        mock_dt.to_local.return_value = torch.ones((2, 2))
        mock_dt.detach.return_value = mock_dt
        with patch(
            "keras.src.backend.torch.core.convert_to_tensor",
            return_value=mock_dt,
        ):
            v = core.Variable(mock_dt, shape=(2, 2))
            mock_dt.to_local.assert_called()
            v._value = MagicMock()
            v._direct_assign(mock_dt)
            mock_dt.to_local.assert_called()

    def test_convert_to_tensor_basics(self):
        with self.assertRaises(ValueError):
            core.convert_to_tensor([1], sparse=True)
        with self.assertRaises(ValueError):
            core.convert_to_tensor([1], ragged=True)
        self.assertAllClose(
            core.convert_to_tensor(core.Variable([1.0, 2.0])), [1.0, 2.0]
        )
        res = core.convert_to_tensor(torch.empty(2, 2, device="meta"))
        self.assertEqual(str(res.device.type), core.get_device())

    def test_convert_to_tensor_distribution(self):
        mock_dist = MagicMock(device_mesh=MagicMock(axis_names=[]))
        orig_get = global_state.get_global_attribute
        with (
            patch(
                "keras.src.backend.common.global_state.get_global_attribute",
                side_effect=lambda a, *args, **k: mock_dist
                if a == "distribution"
                else orig_get(a, *args, **k),
            ),
            patch(
                "keras.src.backend.torch.distribution_lib.distribute_tensor",
                return_value=torch.randn(2, 2),
            ),
        ):
            self.assertIsInstance(
                core.convert_to_tensor(torch.ones((2, 2))), torch.Tensor
            )
        # DTensor branch
        mock_dt = MagicMock(
            spec=DTensor, device=torch.device("cpu"), is_meta=False
        )
        mock_dt.to.return_value = mock_dt
        with (
            patch(
                "keras.src.backend.common.global_state.get_global_attribute",
                return_value=mock_dist,
            ),
            patch("keras.src.backend.torch.core.is_tensor", return_value=True),
        ):
            self.assertIs(core.convert_to_tensor(mock_dt), mock_dt)

    def test_convert_to_numpy_basics(self):
        import ml_dtypes

        self.assertEqual(
            core.convert_to_numpy(
                torch.tensor([1.0], dtype=torch.bfloat16)
            ).dtype,
            ml_dtypes.bfloat16,
        )
        self.assertAllClose(
            core.convert_to_numpy([torch.tensor(1.0), torch.tensor(2.0)]),
            [1.0, 2.0],
        )
        # DTensor
        mock_dt = MagicMock(
            spec=DTensor,
            requires_grad=False,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        mock_dt.full_tensor.return_value = torch.ones((2, 2), device="cpu")
        self.assertAllClose(core.convert_to_numpy(mock_dt), np.ones((2, 2)))
