import os

import numpy as np
import pytest
import torch
import torch.distributed as dist

# The library to be tested
from keras.src.backend import distribution_lib

# We need the Keras API classes to create layouts for testing the backend
# conversion and distribution functions.
from keras.src.distribution import DeviceMesh
from keras.src.distribution import TensorLayout


@pytest.fixture(scope="session", autouse=True)
def setup_torch_distributed():
    """
    A fixture to initialize the distributed process group if not already done.
    This allows the test file to be run directly with `pytest` for single-process
    checks, while also working correctly when launched with `torchrun`.
    """
    if not dist.is_available() or dist.is_initialized():
        return

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")  # A default free port
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    dist.init_process_group(backend="gloo")


@pytest.mark.skipif(
    not torch.distributed.is_available(),
    reason="PyTorch distributed components are not available.",
)
class TestTorchDistributionLibLive:
    """
    Tests for the Torch distribution library without using mocks.
    These tests will reflect the capabilities of the environment they are run in.
    """

    ## Section 1: Device and Process Info
    # ------------------------------------

    def test_get_device_count(self):
        """Tests the get_device_count helper against the runtime environment."""
        assert distribution_lib.get_device_count("cpu") == 1

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            assert distribution_lib.get_device_count("gpu") == gpu_count
            assert distribution_lib.get_device_count("cuda") == gpu_count
        else:
            assert distribution_lib.get_device_count("gpu") == 0

        if torch.cuda.is_available():
            assert (
                distribution_lib.get_device_count() == torch.cuda.device_count()
            )
        elif (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ):
            assert distribution_lib.get_device_count() == 1
        else:
            assert distribution_lib.get_device_count() == 1

    def test_device_listing_and_info(self):
        """Tests device discovery functions against the runtime environment."""
        # Test basic listing based on actual hardware
        if torch.cuda.is_available():
            gpu_devices = distribution_lib.list_devices("gpu")
            assert len(gpu_devices) == torch.cuda.device_count()
            assert gpu_devices[0] == "cuda:0"
        else:
            assert distribution_lib.list_devices("gpu") == []

        cpu_devices = distribution_lib.list_devices("cpu")
        assert cpu_devices == ["cpu:0"]

        # Test error handling for invalid device types
        with pytest.raises(ValueError, match="Unknown device type"):
            distribution_lib.list_devices("unsupported_device")

    def test_device_helpers(self):
        """Tests validation, backend, and memory info functions."""
        device_str = "cpu:0"
        if torch.cuda.is_available():
            device_str = "cuda:0"

        # Test validation
        assert distribution_lib.validate_device_placement(device_str) is True
        assert distribution_lib.validate_device_placement("invalid:0") is False

        # Test backend retrieval
        assert distribution_lib.get_device_backend("cpu") == "torch"
        assert distribution_lib.get_device_backend("gpu") == "torch"

        # Test memory info retrieval
        mem_info = distribution_lib.get_device_memory_info(device_str)
        assert mem_info is not None
        assert "type" in mem_info
        assert mem_info["index"] == 0

    def test_process_discovery(self):
        """Tests process_id and num_processes in the live environment."""
        # This will report the actual rank and world size
        rank = distribution_lib.process_id()
        world_size = distribution_lib.num_processes()

        if dist.is_initialized():
            assert rank == dist.get_rank()
            assert world_size == dist.get_world_size()
        else:
            assert rank == 0
            assert world_size == 1

    ## Section 2: Backend Object Conversion
    # --------------------------------------

    def test_backend_conversions(self):
        """Tests the conversion of Keras objects to Torch backend objects."""
        # Use available devices from the library itself
        world_size = distribution_lib.num_processes()
        if world_size < 2:
            pytest.skip(
                "Skipping conversion tests in a single-process environment."
            )

        devices = [f"cpu:{i}" for i in range(world_size)]
        shape = (world_size,)
        axis_names = ("data",)
        keras_mesh = DeviceMesh(shape, axis_names, devices)

        # Test _to_backend_mesh
        torch_mesh = distribution_lib._to_backend_mesh(keras_mesh)
        assert isinstance(torch_mesh, dist.DeviceMesh)
        assert torch_mesh.mesh.shape == shape

        # Test _to_backend_layout
        keras_layout = TensorLayout(axes=("data",), device_mesh=keras_mesh)
        placements = distribution_lib._to_backend_layout(keras_layout)
        assert isinstance(placements[0], dist.Shard)

        keras_layout_replicated = TensorLayout(
            axes=(None,), device_mesh=keras_mesh
        )
        placements_replicated = distribution_lib._to_backend_layout(
            keras_layout_replicated
        )
        assert isinstance(placements_replicated[0], dist.Replicate)

    ## Section 3: Tensor Distribution
    # --------------------------------

    def test_tensor_distribution(self):
        """Tests the distribution of a tensor into a DTensor."""
        if not dist.is_initialized() or distribution_lib.num_processes() < 2:
            pytest.skip(
                "Tensor distribution test requires a multi-process environment."
            )

        world_size = distribution_lib.num_processes()
        devices = np.arange(world_size)
        keras_mesh = DeviceMesh((world_size,), ("batch",), devices)
        keras_layout = TensorLayout(("batch", None), keras_mesh)

        # This tensor exists on all processes
        local_tensor = torch.randn((10, 20))

        # Test distribute_tensor
        dtensor = distribution_lib.distribute_tensor(local_tensor, keras_layout)
        assert isinstance(dtensor, torch.distributed.dtensor.DTensor)
        assert dtensor.device_mesh.mesh.shape == (world_size,)
        assert isinstance(dtensor.placements[0], dist.Shard)

        # Test distribute_variable (which calls distribute_tensor)
        dvariable = distribution_lib.distribute_variable(
            local_tensor, keras_layout
        )
        assert isinstance(dvariable, torch.distributed.dtensor.DTensor)

    def test_distribute_data_input(self):
        """Tests the `from_local` logic for distributing input data."""
        if not dist.is_initialized() or distribution_lib.num_processes() < 2:
            pytest.skip(
                "Input distribution test requires a multi-process environment."
            )

        world_size = distribution_lib.num_processes()
        devices = np.arange(world_size)
        keras_mesh = DeviceMesh((world_size,), ("batch",), devices)
        # Shard the first dimension ('batch') across all devices
        keras_layout = TensorLayout(("batch", None), keras_mesh)

        # Each process has a local shard of the data
        per_process_batch = torch.ones((8, 16))

        # Distribute the local shards to form a global DTensor
        global_batch = distribution_lib.distribute_data_input(
            per_process_batch, keras_layout, batch_dim_name="batch"
        )

        assert isinstance(global_batch, torch.distributed.dtensor.DTensor)
        # The global shape should be the local shape scaled by the number of devices
        # along the sharded dimension.
        assert global_batch.shape == (world_size * 8, 16)
