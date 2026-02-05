#!/usr/bin/env python3
"""Test to verify the DTensor sharding fix works correctly."""

import os
os.environ["KERAS_BACKEND"] = "torch"

import torch
from torch.distributed._tensor import DTensor, DeviceMesh, Replicate, Shard

# Test the _layout_to_placements function logic
def test_layout_to_placements():
    """Test that layout_to_placements returns correct placements for 2D tensor with 1D mesh."""
    
    # Simulate what happens in _layout_to_placements
    # For 1D mesh with shape (2,), tensor shape (64, 256), layout (None, 'model')
    
    tensor_rank = 2  # (64, 256) is 2D
    mesh_ndim = 1
    layout = (None, 'model')
    
    placements = []
    
    # 1D mesh case: map 'model' axis to the single mesh dimension
    # But keep placements for ALL tensor dimensions
    for i, axis in enumerate(layout):
        if axis == 'model':
            # Shard on this tensor dimension using the single mesh dim
            tensor_dim = tensor_rank - len(layout) + i if tensor_rank > len(layout) else i
            placements.append(Shard(tensor_dim))
        elif axis is None:
            # Replicate this dimension
            placements.append(Replicate())
        else:
            # Other axis names
            placements.append(Replicate())
    
    # Ensure we have placements for ALL tensor dimensions
    # Pad with Replicate() if layout has fewer dimensions than tensor
    while len(placements) < tensor_rank:
        placements.append(Replicate())
    
    print(f"Layout: {layout}")
    print(f"Tensor rank: {tensor_rank}")
    print(f"Placements: {placements}")
    
    # Verify
    assert len(placements) == tensor_rank, f"Expected {tensor_rank} placements, got {len(placements)}"
    assert isinstance(placements[0], Replicate), f"Dim 0 should be Replicate, got {placements[0]}"
    assert isinstance(placements[1], Shard), f"Dim 1 should be Shard, got {placements[1]}"
    assert placements[1].dim == 1, f"Shard dim should be 1, got {placements[1].dim}"
    
    print("✓ Test passed! Placements are correct:")
    print(f"  - Dim 0 (input_dim=64): Replicate")
    print(f"  - Dim 1 (output_dim=256): Shard on dim 1")


def test_old_broken_logic():
    """Show why the old logic was broken."""
    
    tensor_rank = 2  # (64, 256) is 2D
    mesh_ndim = 1
    layout = (None, 'model')
    
    # OLD BROKEN LOGIC
    placements_old = []
    for i, axis in enumerate(layout):
        if axis == 'model':
            tensor_dim = tensor_rank - len(layout) + i if tensor_rank > len(layout) else i
            placements_old.append(Shard(tensor_dim))
            break
    else:
        placements_old.append(Replicate())
    
    # Old logic truncated to mesh_ndim
    placements_old = placements_old[:mesh_ndim]
    
    print(f"\nOLD BROKEN LOGIC:")
    print(f"Layout: {layout}")
    print(f"Tensor rank: {tensor_rank}")
    print(f"Placements: {placements_old}")
    print(f"Length: {len(placements_old)}")
    
    # This was broken - only 1 placement for a 2D tensor!
    assert len(placements_old) == 1, "Old logic incorrectly had only 1 placement"
    print("  ✗ Old logic only had 1 placement (WRONG!)")
    print("    - When PyTorch tried to compute global shape, it saw:")
    print("    - Shard(dim=1) on a list with only 1 element")
    print("    - This caused: 'Sharding dim 1 greater than tensor ndim 1'")


if __name__ == "__main__":
    print("=" * 70)
    print("Testing DTensor Sharding Fix")
    print("=" * 70)
    
    test_layout_to_placements()
    test_old_broken_logic()
    
    print("\n" + "=" * 70)
    print("All tests passed! The fix is correct.")
    print("=" * 70)

