#!/usr/bin/env python3
"""
Test script to verify DTensor sharding fix.

This tests that the _axis_names_to_placements and _layout_to_placements
functions correctly return placements matching device_mesh.ndim.
"""

import os
# Must be set before any other imports
os.environ["KERAS_BACKEND"] = "torch"

import torch
from torch.distributed._tensor import DeviceMesh, Replicate, Shard

# Import the fixed functions directly
import sys
sys.path.insert(0, '/Users/suhanaaa/keras')

from keras.src.backend.torch.distribution_lib import (
    _axis_names_to_placements,
    _layout_to_placements,
    _to_backend_mesh
)
from keras.src.distribution import DeviceMesh as KerasDeviceMesh


def test_axis_names_to_placements_1d():
    """Test _axis_names_to_placements with 1D mesh."""
    print("Test 1: _axis_names_to_placements with 1D mesh")
    
    # Create a 1D DeviceMesh
    mesh_tensor = torch.tensor([0, 1], dtype=torch.int64)
    device_mesh = DeviceMesh(
        device_type="cuda",
        mesh=mesh_tensor,
        mesh_dim_names=["model"]
    )
    
    # Test with 'model' axis
    placements = _axis_names_to_placements(('model',), device_mesh)
    print(f"  axis_names=('model',): {placements}")
    assert len(placements) == device_mesh.mesh.ndim == 1, \
        f"Expected 1 placement, got {len(placements)}"
    assert isinstance(placements[0], Shard), \
        f"Expected Shard placement, got {placements[0]}"
    
    # Test with None axis
    placements = _axis_names_to_placements((None,), device_mesh)
    print(f"  axis_names=(None,): {placements}")
    assert len(placements) == 1
    assert isinstance(placements[0], Replicate)
    
    # Test with empty tuple
    placements = _axis_names_to_placements((), device_mesh)
    print(f"  axis_names=(): {placements}")
    assert len(placements) == 1
    assert isinstance(placements[0], Replicate)
    
    print("  ✓ PASSED\n")


def test_layout_to_placements_1d():
    """Test _layout_to_placements with 1D mesh."""
    print("Test 2: _layout_to_placements with 1D mesh")
    
    # Create a 1D DeviceMesh
    mesh_tensor = torch.tensor([0, 1], dtype=torch.int64)
    device_mesh = DeviceMesh(
        device_type="cuda",
        mesh=mesh_tensor,
        mesh_dim_names=["model"]
    )
    
    # Create a sample tensor (e.g., weight matrix)
    tensor = torch.randn(256, 512)  # 2D tensor
    
    # Test with (None, 'model') layout - should shard on dim 1
    layout = (None, 'model')
    placements = _layout_to_placements(layout, tensor, device_mesh)
    print(f"  layout=(None, 'model'), tensor shape={tensor.shape}: {placements}")
    assert len(placements) == device_mesh.mesh.ndim == 1, \
        f"Expected 1 placement, got {len(placements)}"
    assert isinstance(placements[0], Shard), \
        f"Expected Shard placement, got {placements[0]}"
    
    # Test with () layout - should replicate
    layout = ()
    placements = _layout_to_placements(layout, tensor, device_mesh)
    print(f"  layout=(), tensor shape={tensor.shape}: {placements}")
    assert len(placements) == 1
    assert isinstance(placements[0], Replicate)
    
    print("  ✓ PASSED\n")


def test_to_backend_mesh():
    """Test _to_backend_mesh with Keras DeviceMesh."""
    print("Test 3: _to_backend_mesh with Keras DeviceMesh")
    
    # Create Keras DeviceMesh
    keras_mesh = KerasDeviceMesh(
        shape=(2,),
        axis_names=["model"],
        devices=["cuda:0", "cuda:1"]
    )
    
    # Convert to PyTorch DeviceMesh
    torch_mesh = _to_backend_mesh(keras_mesh)
    print(f"  Keras mesh shape: {keras_mesh.shape}")
    print(f"  Torch mesh shape: {torch_mesh.mesh.shape}")
    print(f"  Torch mesh ndim: {torch_mesh.mesh.ndim}")
    
    assert torch_mesh.mesh.ndim == 1, \
        f"Expected 1D mesh, got ndim={torch_mesh.mesh.ndim}"
    
    print("  ✓ PASSED\n")


def test_end_to_end():
    """Test end-to-end sharding with the test script's setup."""
    print("Test 4: End-to-end sharding test")
    
    # Setup similar to kaggle_hybrid_dp_mp_actual_sharding.py
    from keras.src.distribution import DeviceMesh, LayoutMap, ModelParallel, initialize
    from keras.src.backend.torch import distribution_lib
    
    initialize()
    
    devices = ["cuda:0", "cuda:1"]
    mesh = DeviceMesh(
        shape=(len(devices),),
        axis_names=["model"],
        devices=devices
    )
    
    layout_map = LayoutMap(mesh)
    layout_map[".*dense.*kernel"] = (None, "model")  # Shard on dim 1
    layout_map[".*dense.*bias"] = ()  # Replicate
    
    strategy = ModelParallel(
        layout_map=layout_map,
        batch_dim_name="data",
        auto_shard_dataset=False
    )
    
    with strategy.scope():
        # Create a simple tensor with the layout
        tensor = torch.randn(256, 512)  # Simulating Dense kernel
        placements = _layout_to_placements((None, 'model'), tensor, _to_backend_mesh(mesh))
        print(f"  Kernel tensor shape: {tensor.shape}")
        print(f"  Placements for (None, 'model'): {placements}")
        
        assert len(placements) == 1, \
            f"Expected 1 placement for 1D mesh, got {len(placements)}"
        
        # Test bias tensor (should replicate)
        bias = torch.randn(512)
        placements_bias = _layout_to_placements((), bias, _to_backend_mesh(mesh))
        print(f"  Bias tensor shape: {bias.shape}")
        print(f"  Placements for (): {placements_bias}")
        
        assert len(placements_bias) == 1
    
    print("  ✓ PASSED\n")


if __name__ == "__main__":
    print("=" * 60)
    print("DTensor Sharding Fix Verification Tests")
    print("=" * 60)
    print()
    
    try:
        test_axis_names_to_placements_1d()
        test_layout_to_placements_1d()
        test_to_backend_mesh()
        test_end_to_end()
        
        print("=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

