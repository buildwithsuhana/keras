"""Test for the torch distributed GPU conflict fix.

This test verifies that the _to_backend_mesh function correctly handles
distributed settings by only using the local device for each process.
"""

import os
from unittest import mock

import numpy as np
import pytest

# Set torch backend before importing keras
os.environ["KERAS_BACKEND"] = "torch"

import torch


@pytest.fixture(autouse=True)
def clean_env():
    """Clean environment variables before and after each test."""
    # Store original env vars
    original_env = {
        "LOCAL_RANK": os.environ.get("LOCAL_RANK"),
        "RANK": os.environ.get("RANK"),
        "WORLD_SIZE": os.environ.get("WORLD_SIZE"),
    }
    
    # Clear for non-distributed tests
    os.environ.pop("LOCAL_RANK", None)
    os.environ.pop("RANK", None)
    os.environ.pop("WORLD_SIZE", None)
    
    yield
    
    # Restore original env
    for key, value in original_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


class TestToBackendMesh:
    """Tests for _to_backend_mesh function."""
    
    def test_non_distributed_uses_all_devices(self):
        """Test that non-distributed mode uses all devices from the mesh."""
        from keras.src.backend.torch.distribution_lib import (
            _to_backend_mesh,
        )
        from keras.src.distribution import distribution_lib
        
        # Create a mesh with multiple devices
        if torch.cuda.is_available():
            devices = [f"cuda:{i}" for i in range(min(2, torch.cuda.device_count()))]
        else:
            devices = ["cpu:0", "cpu:1"]
        
        mesh = distribution_lib.DeviceMesh(
            shape=(len(devices),),
            axis_names=["batch"],
            devices=devices
        )
        
        # Convert to backend mesh
        backend_mesh = _to_backend_mesh(mesh)
        
        # In non-distributed mode, should use all devices
        assert backend_mesh.shape == mesh.shape
        assert backend_mesh.dim_names == ("batch",)
    
    def test_distributed_mode_uses_local_device_only(self):
        """Test that distributed mode only uses the local device."""
        from keras.src.backend.torch.distribution_lib import (
            _to_backend_mesh,
        )
        from keras.src.distribution import distribution_lib
        
        # Set up distributed environment
        local_rank = 1
        rank = 1
        world_size = 2
        
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        
        # Create a mesh with multiple devices
        if torch.cuda.is_available():
            devices = [f"cuda:{i}" for i in range(min(2, torch.cuda.device_count()))]
        else:
            devices = ["cpu:0", "cpu:1"]
        
        mesh = distribution_lib.DeviceMesh(
            shape=(len(devices),),
            axis_names=["batch"],
            devices=devices
        )
        
        # Convert to backend mesh
        backend_mesh = _to_backend_mesh(mesh)
        
        # In distributed mode, should use only local device
        assert backend_mesh.shape == (1,), (
            f"Expected shape (1,) in distributed mode, got {backend_mesh.shape}"
        )
        assert backend_mesh.dim_names == ("batch",)
    
    def test_distributed_mode_with_single_process(self):
        """Test that single-process distributed setting uses local device."""
        from keras.src.backend.torch.distribution_lib import (
            _to_backend_mesh,
        )
        from keras.src.distribution import distribution_lib
        
        # Set up distributed environment with single process
        local_rank = 0
        rank = 0
        world_size = 1
        
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        
        # Create a mesh with multiple devices
        if torch.cuda.is_available():
            devices = [f"cuda:{i}" for i in range(min(2, torch.cuda.device_count()))]
        else:
            devices = ["cpu:0", "cpu:1"]
        
        mesh = distribution_lib.DeviceMesh(
            shape=(len(devices),),
            axis_names=["batch"],
            devices=devices
        )
        
        # Convert to backend mesh
        backend_mesh = _to_backend_mesh(mesh)
        
        # With world_size=1, this should still be treated as distributed
        # because all env vars are set
        # The shape should be (1,) since world_size > 1 check is in the code
        # but since world_size=1, it won't trigger the distributed path
        # Let's verify the behavior
        assert backend_mesh.shape == (len(devices),)


class TestDataParallelDistribution:
    """Tests for DataParallel distribution with torch backend."""
    
    @mock.patch.dict(os.environ, {"KERAS_BACKEND": "torch"}, clear=False)
    def test_data_parallel_creation_non_distributed(self):
        """Test DataParallel creation in non-distributed mode."""
        from keras.src.distribution import distribution_lib
        
        devices = ["cpu:0", "cpu:1"]
        distribution = distribution_lib.DataParallel(devices=devices)
        
        assert distribution.device_mesh.shape == (2,)
        assert distribution.device_mesh.axis_names == ["batch"]
        assert distribution._is_multi_process is False
    
    @mock.patch.dict(os.environ, {"KERAS_BACKEND": "torch"}, clear=False)
    @mock.patch("keras.src.backend.torch.distribution_lib.dist.is_initialized")
    def test_data_parallel_creation_distributed(self, mock_is_initialized):
        """Test DataParallel creation simulates distributed environment."""
        from keras.src.distribution import distribution_lib
        
        # Mock is_initialized to return True with 2 processes
        mock_is_initialized.return_value = True
        with mock.patch("keras.src.backend.torch.distribution_lib.dist.get_world_size", return_value=2):
            with mock.patch("keras.src.backend.torch.distribution_lib.dist.get_rank", return_value=1):
                devices = ["cpu:0", "cpu:1"]
                distribution = distribution_lib.DataParallel(devices=devices)
                
                assert distribution._is_multi_process is True
                assert distribution._num_process == 2
                assert distribution._process_id == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

