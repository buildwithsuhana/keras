"""Test DTensor integration for Keras Torch backend distribution."""

import os
import re
import sys

# Set up paths for testing
sys.path.insert(0, '/Users/suhanaaa/keras')

import torch
import torch.distributed as dist

# Test imports
def test_dtensor_availability():
    """Test if DTensor is available in PyTorch."""
    print("Testing DTensor availability...")
    
    if hasattr(torch.distributed, "tensor"):
        print("✓ DTensor is available")
        from torch.distributed.tensor import (
            distribute_tensor,
            DeviceMesh,
            Shard,
            Replicate,
            Partial,
        )
        return True
    else:
        print("✗ DTensor is NOT available - will use fallback")
        return False


def test_path_adaptor():
    """Test the path adapter for Keras/PyTorch naming conventions."""
    print("\nTesting path adapter...")
    
    from keras.src.backend.torch.distribution_lib import (
        _adapt_path,
        _match_layout_map_key,
    )
    
    # Test _adapt_path
    path_keras = "dense/kernel"
    path_torch = "dense.weight"
    
    adapted = _adapt_path(path_keras)
    assert adapted["keras"] == path_keras
    assert adapted["torch"] == path_torch
    print(f"✓ _adapt_path works: {path_keras} -> {adapted}")
    
    # Test _match_layout_map_key with LayoutMap-like dict
    class SimpleLayoutMap(dict):
        """Simple LayoutMap for testing."""
        pass
    
    layout_map = SimpleLayoutMap()
    from keras.src.distribution import TensorLayout, DeviceMesh
    
    devices = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
    mesh = DeviceMesh(shape=(4,), axis_names=["batch"], devices=devices)
    
    # Add some layouts with both formats
    layout_map["dense/kernel"] = TensorLayout((None, "batch"), mesh)
    layout_map["conv.*bias"] = TensorLayout(("batch",), mesh)
    
    # Test matching with Keras format
    result = _match_layout_map_key("dense/kernel", layout_map)
    assert result is not None
    print("✓ Path adapter matches Keras format (dense/kernel)")
    
    # Test matching with regex
    result = _match_layout_map_key("conv2d_1/bias", layout_map)
    assert result is not None
    print("✓ Path adapter matches regex pattern (conv.*bias)")
    
    # Test no match
    result = _match_layout_map_key("unknown/layer", layout_map)
    assert result is None
    print("✓ Path adapter returns None for non-matching paths")
    
    return True


def test_dtensor_converter_functions():
    """Test DTensor converter functions."""
    print("\nTesting DTensor converter functions...")
    
    from keras.src.backend.torch.distribution_lib import (
        _is_dtensor_available,
        _to_dtensor_device,
    )
    
    # Test availability check
    dtensor_available = _is_dtensor_available()
    print(f"✓ DTensor availability: {dtensor_available}")
    
    # Test device conversion
    device = _to_dtensor_device("cuda:0")
    assert str(device).startswith("cuda")
    print(f"✓ Device conversion: cuda:0 -> {device}")
    
    device = _to_dtensor_device("cpu")
    assert str(device) == "cpu"
    print(f"✓ Device conversion: cpu -> {device}")
    
    return True


def test_fallback_distribution():
    """Test fallback distribution when DTensor is not available."""
    print("\nTesting fallback distribution...")
    
    from keras.src.distribution import TensorLayout, DeviceMesh
    from keras.src.backend.torch.distribution_lib import (
        distribute_tensor,
        distribute_variable,
        _distribute_tensor_fallback,
    )
    
    # Create a simple layout without DTensor
    devices = ["cpu:0"]  # Single device for testing
    mesh = DeviceMesh(shape=(1,), axis_names=["batch"], devices=devices)
    layout = TensorLayout((None, "batch"), mesh)
    
    # Test distribute_tensor
    tensor = torch.randn(8, 16)
    result = distribute_tensor(tensor, layout)
    
    # Should return the original tensor (replicated)
    assert result.shape == tensor.shape
    print("✓ distribute_tensor fallback works")
    
    # Test distribute_variable
    var = torch.nn.Parameter(torch.randn(8, 16))
    result = distribute_variable(var, layout)
    
    assert result.shape == var.shape
    print("✓ distribute_variable fallback works")
    
    return True


def test_all_gather_variable():
    """Test all_gather_variable function."""
    print("\nTesting all_gather_variable...")
    
    from keras.src.backend.torch.distribution_lib import (
        all_gather_variable,
    )
    
    # Test with a non-sharded tensor
    tensor = torch.randn(8, 16)
    result = all_gather_variable(tensor)
    
    # Should return the same tensor
    assert result.shape == tensor.shape
    print("✓ all_gather_variable works with non-sharded tensor")
    
    # Test with a manually marked sharded tensor
    sharded = torch.randn(4, 16)
    sharded._is_sharded = True
    sharded._full_shape = (8, 16)
    sharded._sharding_axis = 0
    
    # Note: This requires distributed to be initialized
    if dist.is_initialized():
        result = all_gather_variable(sharded)
        assert result.shape == (8, 16)
        print("✓ all_gather_variable works with sharded tensor")
    else:
        print("⚠ Skipping distributed test (not initialized)")
    
    return True


def test_collective_operations():
    """Test collective operations (all_reduce, all_gather, broadcast)."""
    print("\nTesting collective operations...")
    
    from keras.src.backend.torch.distribution_lib import (
        all_reduce,
        all_gather,
        broadcast,
    )
    
    # Test all_reduce
    tensor = torch.randn(8, 16)
    result = all_reduce(tensor, op="sum")
    assert result.shape == tensor.shape
    print("✓ all_reduce works")
    
    # Test all_gather
    tensor = torch.randn(8, 16)
    result = all_gather(tensor, axis=0)
    assert result.shape == tensor.shape
    print("✓ all_gather works")
    
    # Test broadcast
    tensor = torch.randn(8, 16)
    result = broadcast(tensor, src=0)
    assert result.shape == tensor.shape
    print("✓ broadcast works")
    
    return True


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("DTensor Integration Tests for Keras Torch Backend")
    print("=" * 60)
    
    tests = [
        ("DTensor Availability", test_dtensor_availability),
        ("Path Adapter", test_path_adaptor),
        ("Converter Functions", test_dtensor_converter_functions),
        ("Fallback Distribution", test_fallback_distribution),
        ("All Gather Variable", test_all_gather_variable),
        ("Collective Operations", test_collective_operations),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ {name} FAILED: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

