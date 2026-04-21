from unittest.mock import MagicMock
from unittest.mock import patch

import ml_dtypes
import numpy as np
import pytest
import torch
from torch.distributed.tensor import DTensor

from keras.src import backend
from keras.src import distribution
from keras.src import testing
from keras.src.backend.common import global_state
from keras.src.backend.torch import core
from keras.src.backend.torch.distribution_lib_test import (
    TorchDistributedTestCase,
)


@pytest.mark.skipif(
    backend.backend() != "torch", reason="Requires Torch backend"
)
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

        # Test bool
        t_bool = core.convert_to_tensor(True)
        self.assertEqual(t_bool.dtype, torch.bool)
        self.assertAllClose(t_bool, True)

        # Test float
        from keras.src.backend.config import floatx

        t_float = core.convert_to_tensor(1.5)
        self.assertEqual(t_float.dtype, core.to_torch_dtype(floatx()))
        self.assertAllClose(t_float, 1.5)

        # Test large int
        x = 2**31
        t_int64 = core.convert_to_tensor(x)
        self.assertEqual(t_int64.dtype, torch.int64)

        # Test meta device
        x_meta = torch.tensor([1.0, 2.0], device="meta")
        t_meta = core.convert_to_tensor(x_meta)
        self.assertEqual(
            t_meta.device.type, torch.device(core.get_device()).type
        )

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

        # Test trainable setter with value is None (already initialized above)
        v3.trainable = False
        self.assertFalse(v3.trainable)
        self.assertFalse(v3.value.requires_grad)

        # Test __array__
        arr = np.array(v3)
        self.assertAllClose(arr, np.ones((2, 2)))
        arr_64 = v3.__array__(dtype="float64")
        self.assertEqual(arr_64.dtype, "float64")

        # Test __torch_function__
        # kwargs=None branch
        result = torch.add(v3, core.convert_to_tensor(1.0))
        self.assertAllClose(result, np.ones((2, 2)) * 2.0)
        # kwargs not None branch
        result = torch.add(v3, alpha=1.0, other=core.convert_to_tensor(1.0))
        self.assertAllClose(result, np.ones((2, 2)) * 2.0)

    def test_compute_output_spec(self):
        from keras.src.backend.common.keras_tensor import KerasTensor

        def fn(x):
            return x + 1

        x = KerasTensor((2, 2), dtype="float32")
        output_spec = core.compute_output_spec(fn, x)
        self.assertIsInstance(output_spec, KerasTensor)
        self.assertEqual(output_spec.shape, (2, 2))

    def test_device_scope(self):
        original_device = core.get_device()
        with core.device_scope("cpu"):
            self.assertEqual(core.get_device(), "cpu")
            with core.device_scope(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            ):
                expected = "cuda:0" if torch.cuda.is_available() else "cpu"
                self.assertEqual(core.get_device(), expected)
            self.assertEqual(core.get_device(), "cpu")
        self.assertEqual(core.get_device(), original_device)

    def test_parse_device_input(self):
        self.assertEqual(core._parse_device_input("cpu"), "cpu")
        self.assertEqual(core._parse_device_input("GPU:0"), "cuda:0")
        with self.assertRaisesRegex(
            ValueError, "Invalid value for argument `device_name`"
        ):
            core._parse_device_input(123)

    def test_to_torch_dtype(self):
        self.assertEqual(core.to_torch_dtype("float32"), torch.float32)
        self.assertEqual(core.to_torch_dtype(torch.float64), torch.float64)
        # Valid Keras dtype but not supported by Torch backend
        with self.assertRaisesRegex(
            ValueError, "Unsupported dtype for PyTorch"
        ):
            core.to_torch_dtype("string")
        # Invalid Keras dtype
        with self.assertRaisesRegex(ValueError, "Invalid dtype: invalid"):
            core.to_torch_dtype("invalid")

    def test_cast(self):
        x = torch.tensor([1.0, 2.0], dtype=torch.float32)
        y = core.cast(x, "float64")
        self.assertEqual(y.dtype, torch.float64)

        v = core.Variable([1, 2], dtype="int32")
        y = core.cast(v, "float32")
        self.assertEqual(y.dtype, torch.float32)

    def test_cond(self):
        self.assertEqual(core.cond(True, lambda: 1, lambda: 2), 1)
        self.assertEqual(core.cond(False, lambda: 1, lambda: 2), 2)

        # Test with meta device
        with core.device_scope("meta"):
            self.assertEqual(core.cond(False, lambda: 1, lambda: 2), 1)

    def test_control_flow(self):
        # test_switch
        branches = [lambda: 10, lambda: 20, lambda: 30]
        self.assertEqual(core.switch(0, branches), 10)
        self.assertEqual(core.switch(5, branches), 30)  # Clamped

        # test_while_loop
        def cond(i, x):
            return i < 3

        def body(i, x):
            return i + 1, x + i

        i, x = core.while_loop(cond, body, [0, 0])
        self.assertEqual(i, 3)
        self.assertEqual(x, 3)

        # test_fori_loop
        def body_fori(i, x):
            return x + i

        x = core.fori_loop(0, 3, body_fori, 0)
        self.assertEqual(x, 3)

    def test_scatter_ops(self):
        # test_scatter
        indices = [[0], [2]]
        values = [10.0, 20.0]
        shape = (3,)
        result = core.scatter(indices, values, shape)
        self.assertAllClose(result, [10.0, 0.0, 20.0])

        # test_scatter_update
        inputs = torch.tensor([1.0, 2.0, 3.0])
        result = core.scatter_update(inputs, indices, values, reduction="add")
        self.assertAllClose(result, [11.0, 2.0, 23.0])

    def test_slice_ops(self):
        # test_slice
        inputs = torch.tensor([[1, 2, 3], [4, 5, 6]])
        result = core.slice(inputs, [0, 1], [2, 2])
        self.assertAllClose(result, [[2, 3], [5, 6]])

        # test_slice_update
        updates = torch.tensor([[10, 20], [30, 40]])
        result = core.slice_update(inputs, [0, 1], updates)
        self.assertAllClose(result, [[1, 10, 20], [4, 30, 40]])

    def test_scans(self):
        # test_scan
        def f_scan(carry, x):
            return carry + x, carry + x

        xs = torch.tensor([1, 2, 3])
        carry, ys = core.scan(f_scan, torch.tensor(0), xs)
        self.assertEqual(carry, 6)
        self.assertAllClose(ys, [1, 3, 6])

        # test_associative_scan
        def f_assoc(a, b):
            return a + b

        elems = torch.tensor([1, 2, 3, 4])
        result = core.associative_scan(f_assoc, elems)
        self.assertAllClose(result, [1, 3, 6, 10])

    def test_custom_gradient_and_remat(self):
        # test_remat
        def f(x):
            return x * x

        f_remat = core.remat(f)
        x = torch.tensor([2.0], requires_grad=True)
        y = f_remat(x)
        y.backward()
        self.assertAllClose(x.grad, [4.0])

        # test_custom_gradient
        @core.custom_gradient
        def my_op(x):
            def grad_fn(*args, upstream):
                return upstream * 2

            return x * 3, grad_fn

        x = torch.tensor([2.0], requires_grad=True)
        y = my_op(x)
        y.backward()
        self.assertAllClose(x.grad, [2.0])

    def test_variable_advanced(self):
        # test_variable_stateless
        from keras.src.backend.common.stateless_scope import StatelessScope

        v = core.Variable([1.0, 2.0])
        mapping = [(v, torch.tensor([3.0, 4.0], device=core.get_device()))]
        with StatelessScope(state_mapping=mapping):
            self.assertAllClose(v.value, [3.0, 4.0])

        # test_variable_uninitialized
        def initializer(shape, dtype):
            return torch.ones(
                shape,
                dtype=core.to_torch_dtype(dtype),
                device=core.get_device(),
            )

        with StatelessScope():
            v_uninit = core.Variable(initializer, shape=(2, 2), dtype="float32")
            self.assertIsNone(v_uninit._value)
            self.assertAllClose(v_uninit.value, [[1.0, 1.0], [1.0, 1.0]])

        # test_variable_requires_grad_mismatch
        # line 144
        p = torch.nn.Parameter(
            torch.tensor(
                [1.0, 2.0], requires_grad=True, device=core.get_device()
            )
        )
        v_mismatch = core.Variable(p, trainable=False)
        self.assertFalse(v_mismatch.value.requires_grad)

        # test_variable_eq_exception
        with patch(
            "keras.src.backend.torch.numpy.equal",
            side_effect=Exception("forced"),
        ):
            self.assertFalse(v == "something")

        # test_variable_symbolic
        with core.device_scope("meta"):
            val = v.value
            self.assertEqual(str(val.device), "meta")

    def test_variable_with_mock_distribution(self):
        # lines 114-120, 127-135, 170-171
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

    def test_torch_function_mixed(self):
        # lines 186-193
        v = core.Variable([1.0], dtype="float32")
        # Mixed args
        res = torch.add(
            v,
            torch.tensor([2.0], device=core.get_device(), dtype=torch.float32),
        )
        self.assertAllClose(res, [3.0])
        # Mixed kwargs
        res = torch.add(
            v,
            alpha=1.0,
            other=torch.tensor(
                [2.0], device=core.get_device(), dtype=torch.float32
            ),
        )
        self.assertAllClose(res, [3.0])

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
                return_value=torch.randn(
                    2, 2, device=core.get_device(), dtype=torch.float32
                ),
            ):
                t = torch.ones(
                    (2, 2), device=core.get_device(), dtype=torch.float32
                )
                res = core.convert_to_tensor(t)
                self.assertIsInstance(res, torch.Tensor)

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

    def test_convert_to_numpy_non_tensor(self):
        # hits branch 327->343 (False branch)
        res = core.convert_to_numpy([1, 2])
        self.assertAllClose(res, [1, 2])


@pytest.mark.skipif(
    backend.backend() != "torch", reason="Requires Torch backend"
)
class TorchCoreDistributedTest(TorchDistributedTestCase):
    """Distributed tests for core.py coverage."""

    @staticmethod
    def _test_variable_initialize_layout_and_distribute(self, rank, world_size):
        from keras.src.backend.torch.core import Variable

        mesh = distribution.DeviceMesh((world_size,), ["model"])
        layout_map = distribution.LayoutMap(mesh)
        layout_map[".*"] = distribution.TensorLayout(["model", None], mesh)
        dist = distribution.ModelParallel(
            layout_map=layout_map, batch_dim_name=None
        )

        with dist.scope():
            # layout=None → triggers _initialize_layout → get_variable_layout
            v = Variable(np.ones((world_size, 2)), dtype="float32")
            self.assertIsInstance(v.value, torch.nn.Parameter)
            value = v.value.data
            self.assertIsInstance(value, DTensor)
            self.assertEqual(tuple(value.shape), (world_size, 2))

    @staticmethod
    def _test_convert_to_tensor_model_parallel_fallback(self, rank, world_size):
        """Test convert_to_tensor non-DTensor tensor → replicated layout."""
        t = torch.ones((4, 4), device=core.get_device(), dtype=torch.float32)
        mesh = distribution.DeviceMesh((world_size,), ["batch"])
        dist = distribution.ModelParallel(
            layout_map=distribution.LayoutMap(mesh), batch_dim_name="batch"
        )

        with dist.scope():
            converted = core.convert_to_tensor(t)
            self.assertIsInstance(converted, DTensor)
            self.assertEqual(tuple(converted.shape), t.shape)
            self.assertTrue(
                all(
                    isinstance(p, torch.distributed.tensor.Replicate)
                    for p in converted.placements
                )
            )

    def test_variable_initialize_layout_and_distribute(self):
        self.run_distributed(
            TorchCoreDistributedTest._test_variable_initialize_layout_and_distribute
        )

    def test_convert_to_tensor_model_parallel_fallback(self):
        self.run_distributed(
            TorchCoreDistributedTest._test_convert_to_tensor_model_parallel_fallback
        )
