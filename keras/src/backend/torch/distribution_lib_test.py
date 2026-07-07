import os
from unittest import mock

import numpy as np
import pytest
import torch
from absl.testing import parameterized

from keras.src import backend
from keras.src import testing
from keras.src.backend.torch import distribution_lib
from keras.src.distribution.distribution_lib import DeviceMesh


@pytest.mark.skipif(backend.backend() != "torch", reason="Requires torch")
class TorchDistributionLibTest(testing.TestCase):
    def set_env(self, key, value):
        old = os.environ.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
        self.addCleanup(
            lambda: os.environ.update({key: old})
            if old is not None
            else os.environ.pop(key, None)
        )

    def tearDown(self):
        super().tearDown()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    @parameterized.parameters(
        ({}, False, None, None, 1),
        ({"WORLD_SIZE": "4"}, False, None, None, 4),
        (
            {
                "MASTER_ADDR": "localhost",
                "MASTER_PORT": "29502",
                "WORLD_SIZE": "1",
                "RANK": "0",
            },
            True,
            None,
            None,
            1,
        ),
        ({}, False, "cuda", {"available": True, "count": 2}, 2),
        ({}, False, "gpu", {"available": True, "count": 2}, 2),
    )
    def test_get_device_count(self, env, init, dtype, cuda, expected):
        for k, v in env.items():
            self.set_env(k, v)
        if init:
            torch.distributed.init_process_group(
                backend="gloo", rank=0, world_size=1
            )

        m_cuda = mock.patch(
            "torch.cuda.is_available",
            return_value=cuda["available"]
            if cuda
            else torch.cuda.is_available(),
        )
        m_cuda_c = mock.patch(
            "torch.cuda.device_count",
            return_value=cuda.get("count", 0) if cuda else 0,
        )

        with m_cuda, m_cuda_c:
            res = distribution_lib.get_device_count(dtype)

            if expected is not None:
                self.assertEqual(res, expected)
            else:
                self.assertEqual(
                    res,
                    torch.cuda.device_count()
                    if torch.cuda.is_available()
                    else 1,
                )

    @parameterized.parameters(
        (None, {}, False, True, None),
        ("cpu", {}, False, False, ["cpu:0"]),
        (
            "gpu",
            {"WORLD_SIZE": "4", "KERAS_TORCH_DEVICE": "gpu"},
            False,
            False,
            ["gpu:0", "gpu:1", "gpu:2", "gpu:3"],
        ),
    )
    def test_list_devices(self, dtype, env, init, default, expected):
        for k, v in env.items():
            self.set_env(k, v)
        if init:
            torch.distributed.init_process_group(
                backend="gloo", rank=0, world_size=1
            )

        # Mock CUDA available for GPU tests to ensure they return world size
        cuda_val = True if dtype == "gpu" or (env.get("WORLD_SIZE")) else False
        with mock.patch("torch.cuda.is_available", return_value=cuda_val):
            if default:
                devices = distribution_lib.list_devices()
                self.assertTrue(
                    any(devices[0].startswith(s) for s in ["gpu:", "cpu:"])
                )
            else:
                self.assertEqual(distribution_lib.list_devices(dtype), expected)

    @parameterized.parameters(
        ({}, False, 1),
        ({"WORLD_SIZE": "4"}, False, 4),
        (
            {
                "MASTER_ADDR": "localhost",
                "MASTER_PORT": "29504",
                "WORLD_SIZE": "1",
                "RANK": "0",
            },
            True,
            1,
        ),
    )
    def test_num_processes_and_id(self, env, init, expected):
        for k, v in env.items():
            self.set_env(k, v)
        if init:
            torch.distributed.init_process_group(
                backend="gloo", rank=0, world_size=1
            )
            self.assertEqual(distribution_lib.num_processes(), expected)
            self.assertEqual(
                distribution_lib.process_id(), int(env.get("RANK", 0))
            )
        else:
            self.assertEqual(distribution_lib.num_processes(), expected)
            self.assertEqual(
                distribution_lib.process_id(), int(env.get("RANK", 0))
            )

    @parameterized.parameters(
        (
            "127.0.0.1:29506",
            1,
            0,
            {"MASTER_ADDR": None, "MASTER_PORT": None},
            {"MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29506"},
            False,
        ),
        (
            "127.0.0.1",
            1,
            0,
            {"MASTER_ADDR": None, "MASTER_PORT": None},
            {"MASTER_ADDR": "127.0.0.1"},
            False,
        ),
        (
            None,
            1,
            0,
            {"MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29500"},
            {"MASTER_ADDR": "127.0.0.1"},
            False,
        ),
        (
            "127.0.0.1:29506",
            1,
            0,
            {
                "MASTER_ADDR": "localhost",
                "MASTER_PORT": "29502",
                "WORLD_SIZE": "1",
                "RANK": "0",
            },
            {},
            True,
        ),  # cuda
    )
    def test_initialize(self, addr, nproc, pid, initial, expected, cuda):
        for k, v in initial.items():
            self.set_env(k, v)

        with mock.patch("torch.distributed.init_process_group") as m_init:
            if cuda:
                with (
                    mock.patch("torch.cuda.is_available", return_value=True),
                    mock.patch("torch.cuda.set_device") as m_set,
                ):
                    distribution_lib.initialize(addr, nproc, pid)
                    m_set.assert_called_once()
                    m_init.assert_called_once()
                    args, kwargs = m_init.call_args
                    self.assertEqual(kwargs["backend"], "nccl")
                    self.assertEqual(kwargs["rank"], 0)
                    self.assertEqual(kwargs["world_size"], 1)
            else:
                distribution_lib.initialize(addr, nproc, pid)
                m_init.assert_called_once()

        for k, v in expected.items():
            self.assertEqual(os.environ.get(k), v)

    @parameterized.parameters(
        ("cpu", False, "cpu"),
        (None, True, "cuda"),
        (None, False, "cpu"),  # fallback
    )
    def test_get_device_type(self, k_dev, cuda, expected):
        self.set_env("KERAS_TORCH_DEVICE", k_dev)
        with mock.patch("torch.cuda.is_available", return_value=cuda):
            self.assertEqual(distribution_lib._get_device_type(), expected)

    @parameterized.parameters(
        ("cpu", {}, False, "cpu", None),
        ("gpu", {}, True, "cuda", None),
        (torch.device("cuda:0"), {}, False, "cuda", 0),
        ("cuda", {"LOCAL_RANK": "2"}, False, "cuda", 2),
        ("cuda", {}, True, "cuda", 0),  # id None
        ("cpu:0", {}, False, "cpu", None),  # ":" and None id
        (torch.device("cpu"), {}, False, "cpu", None),  # already device
    )
    def test_to_backend_device(self, inp, env, cuda, etype, eidx):
        for k, v in env.items():
            self.set_env(k, v)
        if (
            isinstance(inp, torch.device)
            and inp.type == "cuda"
            and not torch.cuda.is_available()
        ):
            self.skipTest("No CUDA")
        with mock.patch("torch.cuda.is_available", return_value=cuda):
            dev = distribution_lib._to_backend_device(inp)
            self.assertEqual(dev.type, etype)
            if eidx is not None:
                self.assertEqual(dev.index, eidx)

    @parameterized.parameters(
        (np.array(["cpu:0", "cpu:1"]).reshape(1, 2), "cpu", False),
        (np.array(["gpu:0", "gpu:1"]).reshape(1, 2), "cuda", True),
    )
    def test_to_backend_mesh(self, devs, etype, cuda):
        mesh = DeviceMesh(shape=(1, 2), axis_names=["x", "y"], devices=devs)
        with (
            mock.patch("torch.cuda.is_available", return_value=cuda),
            mock.patch("torch.distributed.device_mesh.DeviceMesh") as m_mesh,
        ):
            distribution_lib._to_backend_mesh(mesh)
            args, kwargs = m_mesh.call_args
            # The first argument is device_type
            self.assertEqual(args[0], etype)
            self.assertEqual(kwargs["mesh_dim_names"], ("x", "y"))
