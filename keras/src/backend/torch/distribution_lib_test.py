import os
import pytest
import torch
import numpy as np
import torch.distributed as dist
import torch.distributed.dtensor

from keras.src.backend import distribution_lib
from keras.src.distribution import DeviceMesh, TensorLayout

@pytest.fixture(scope="session", autouse=True)
def setup_torch_distributed():
    """Initializes the distributed process group if not already done."""
    if not dist.is_available() or dist.is_initialized():
        return

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)


@pytest.mark.skipif(
    not torch.distributed.is_available(),
    reason="PyTorch distributed components are not available.",
)
class TestTorchDistributionLibLive:
    """Tests for the Torch distribution library matching the provided implementation."""

    def test_get_device_count(self):
        """Tests the get_device_count helper."""
        assert distribution_lib.get_device_count("cpu") == 1
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            assert distribution_lib.get_device_count("gpu") == gpu_count
            assert distribution_lib.get_device_count("cuda") == gpu_count
        
        expected_default = torch.cuda.device_count() if torch.cuda.is_available() else 1
        assert distribution_lib.get_device_count() == expected_default

    def test_device_listing_and_info(self):
        """Tests device discovery and info retrieval."""
        if torch.cuda.is_available():
            gpu_devices = distribution_lib.list_devices("gpu")
            assert len(gpu_devices) > 0
            assert "cuda:0" in gpu_devices
            
            info = distribution_lib.get_device_info("cuda:0")
            assert info["type"] == "GPU"
            assert info["index"] == 0
        
        cpu_devices = distribution_lib.list_devices("cpu")
        assert cpu_devices == ["cpu:0"]
        
        info_cpu = distribution_lib.get_device_info("cpu:0")
        assert info_cpu["type"] == "CPU"

    def test_auto_configure_tensor_parallel(self):
        """Tests the auto-configuration utility."""
        config = distribution_lib.auto_configure_tensor_parallel(world_size=1)
        assert "devices" in config
        assert config["world_size"] == 1
        assert config["backend"] == "torch"

    def test_backend_conversions(self):
        """Tests conversion logic for Mesh and Layout."""
        devices = np.array(["cpu:0"])
        keras_mesh = DeviceMesh(shape=(1,), axis_names=("data",), devices=devices)

        torch_mesh = distribution_lib._to_backend_mesh(keras_mesh)
        assert isinstance(torch_mesh, dist.DeviceMesh)
        assert torch_mesh.device_type in ("cuda", "cpu")

        keras_layout = TensorLayout(axes=("data",), device_mesh=keras_mesh)
        placements = distribution_lib._to_backend_layout(keras_layout)
        assert isinstance(placements[0], dist.Shard)
        assert placements[0].dim == 0

        keras_layout_replicated = TensorLayout(axes=(None,), device_mesh=keras_mesh)
        placements_rep = distribution_lib._to_backend_layout(keras_layout_replicated)
        assert isinstance(placements_rep[0], dist.Replicate)

    def test_tensor_distribution(self):
        """Tests distribute_tensor logic using DTensor."""
        devices = np.array(["cpu:0"])
        keras_mesh = DeviceMesh(shape=(1,), axis_names=("batch",), devices=devices)
        keras_layout = TensorLayout(("batch",), keras_mesh)

        local_tensor = torch.ones((4, 4))
        
        dtensor = distribution_lib.distribute_tensor(local_tensor, keras_layout)
        
        assert isinstance(dtensor, torch.distributed.dtensor.DTensor)
        assert dtensor.shape == (4, 4)

    def test_all_reduce_logic(self):
        """Tests the all_reduce wrapper logic."""
        x = torch.tensor([1.0, 2.0])
        result = distribution_lib.all_reduce(x, op="sum")
        
        assert torch.is_tensor(result)
        if dist.is_initialized():
            assert torch.allclose(result, x)

    def test_invalid_layout_handling(self):
        """Verifies that the lib raises errors for invalid inputs as coded."""
        with pytest.raises(ValueError, match="is not found in the device mesh axes"):
            devices = np.array(["cpu:0"])
            keras_mesh = DeviceMesh(shape=(1,), axis_names=("data",), devices=devices)
            keras_layout = TensorLayout(axes=("invalid_axis",), device_mesh=keras_mesh)
            distribution_lib._to_backend_layout(keras_layout)

        with pytest.raises(ValueError, match="instance is required"):
            distribution_lib.distribute_data_input(torch.ones(1), layout=[1, 2, 3], batch_dim_name="batch")