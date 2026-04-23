import ml_dtypes
import numpy as np
import pytest
import torch
from torch.distributed.tensor import DTensor
from torch.distributed.tensor import Replicate

import keras
from keras.src import backend
from keras.src import testing
from keras.src.backend.torch import core


@pytest.mark.skipif(
    backend.backend() != "torch", reason="Torch specific tests."
)
class TorchCoreTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                "gloo", rank=0, world_size=1, init_method="tcp://127.0.0.1:0"
            )

    def test_distribution_integration(self):
        mesh = keras.distribution.DeviceMesh((1,), ["data"], ["cpu:0"])
        layout_map = keras.distribution.LayoutMap(mesh)
        layout_map[".*"] = (None,)
        dist = keras.distribution.ModelParallel(
            layout_map=layout_map, batch_dim_name="data"
        )
        with dist.scope():
            v = backend.Variable(np.ones((2, 2), dtype="float32"))
            self.assertIsInstance(v.value.data, DTensor)
            v._direct_assign(np.zeros((2, 2), dtype="float32"))
            self.assertIsInstance(v.value.data, DTensor)

            t = backend.convert_to_tensor(
                torch.ones((2, 2), dtype=torch.float32)
            )
            self.assertNotIsInstance(t, DTensor)
            self.assertAllClose(backend.convert_to_tensor(t), t)
            stack_result = backend.convert_to_tensor([t, t])
            self.assertNotIsInstance(stack_result, DTensor)

            v_tensor = v.value.data
            result = v_tensor + 1
            self.assertIsInstance(result, DTensor)

            spec = backend.compute_output_spec(
                lambda x: x + 1,
                keras.KerasTensor(shape=(2, 2), dtype="float32"),
            )
            self.assertEqual(spec.shape, (2, 2))

    def test_variable_with_distributed_tensor(self):
        mesh = keras.distribution.DeviceMesh((1,), ["data"], ["cpu:0"])
        dt = torch.distributed.tensor.distribute_tensor(
            torch.ones(2, 2, dtype=torch.float32),
            mesh.backend_mesh,
            [Replicate()],
        )
        v = backend.Variable(dt)
        self.assertNotIsInstance(v.value.data, DTensor)
        self.assertAllClose(v.value, torch.ones(2, 2))

    def test_convert_to_tensor_basics(self):
        for arg in [{"sparse": True}, {"ragged": True}]:
            with self.assertRaises(ValueError):
                backend.convert_to_tensor([1], **arg)
        self.assertAllClose(
            backend.convert_to_tensor(backend.Variable([1.0, 2.0])), [1.0, 2.0]
        )
        t = backend.convert_to_tensor(
            torch.empty(2, 2, device="meta", dtype=torch.float32)
        )
        self.assertEqual(str(t.device.type), core.get_device())

    def test_convert_to_numpy_basics(self):
        self.assertEqual(
            backend.convert_to_numpy(
                torch.tensor([1.0], dtype=torch.bfloat16)
            ).dtype,
            ml_dtypes.bfloat16,
        )
        self.assertAllClose(
            backend.convert_to_numpy([torch.tensor(1.0), torch.tensor(2.0)]),
            [1.0, 2.0],
        )
        mesh = keras.distribution.DeviceMesh((1,), ["data"], ["cpu:0"])
        dt = torch.distributed.tensor.distribute_tensor(
            torch.ones(2, 2, dtype=torch.float32),
            mesh.backend_mesh,
            [Replicate()],
        )
        self.assertAllClose(backend.convert_to_numpy(dt), np.ones((2, 2)))
