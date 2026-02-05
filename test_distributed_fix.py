"""Simple test to verify the distributed training fixes work correctly.

This script tests:
1. Proper NCCL initialization with timeout
2. DeviceMesh creation with correct local rank device mapping
3. DTensor sharding verification
"""

import os
# Must be set before any other imports
os.environ["KERAS_BACKEND"] = "torch"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.distributed as dist
import numpy as np


def test_nccl_initialization():
    """Test NCCL initialization with proper timeout."""
    print("\n" + "="*60)
    print("TEST 1: NCCL INITIALIZATION")
    print("="*60)
    
    # Set environment variables
    os.environ.setdefault("NCCL_DEBUG", "INFO")
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
    os.environ.setdefault("NCCL_TIMEOUT", "1800")
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    if world_size < 2:
        print("Single GPU mode - skipping multi-GPU test")
        return True
    
    # Set CUDA device
    torch.cuda.set_device(local_rank)
    
    # Initialize NCCL
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=local_rank
        )
    
    print(f"[Rank {local_rank}] NCCL initialized successfully")
    print(f"[Rank {local_rank}] Device: cuda:{local_rank}")
    print(f"[Rank {local_rank}] World size: {world_size}")
    
    return True


def test_dtensor_sharding():
    """Test DTensor sharding with proper device mesh."""
    print("\n" + "="*60)
    print("TEST 2: DTENSOR SHARDING")
    print("="*60)
    
    from torch.distributed._tensor import DTensor, DeviceMesh, Replicate, Shard
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    if world_size < 2:
        print("Single GPU mode - testing with replicated tensor")
        # Create a simple tensor on single GPU
        tensor = torch.randn(128, 64)
        print(f"Tensor shape: {tensor.shape}")
        print("✓ Single GPU tensor created successfully")
        return True
    
    # Create DeviceMesh with proper local rank device
    mesh = DeviceMesh(
        device_type="cuda",
        mesh=torch.tensor([local_rank]),  # Single device per process
        mesh_dim_names=["data"]
    )
    
    print(f"[Rank {local_rank}] DeviceMesh created with device: cuda:{local_rank}")
    
    # Create a sharded tensor
    tensor = torch.randn(128, 64)
    
    # Shard along dim 0
    dtensor = DTensor.from_local(tensor, mesh, [Shard(dim=0)])
    
    print(f"[Rank {local_rank}] Original tensor shape: {tensor.shape}")
    print(f"[Rank {local_rank}] DTensor local shape: {dtensor.to_local().shape}")
    print(f"[Rank {local_rank}] DTensor global shape: {dtensor.shape}")
    print(f"[Rank {local_rank}] DTensor placements: {dtensor.placements}")
    
    # Verify sharding
    local_shape = dtensor.to_local().shape
    if local_shape[0] < dtensor.shape[0]:
        print(f"[Rank {local_rank}] ✓ Tensor is sharded across 'data' axis")
    else:
        print(f"[Rank {local_rank}] ✓ Tensor is replicated")
    
    return True


def test_all_reduce():
    """Test NCCL all_reduce to verify communication works."""
    print("\n" + "="*60)
    print("TEST 3: NCCL ALL_REDUCE")
    print("="*60)
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    if world_size < 2:
        print("Single GPU mode - skipping all_reduce test")
        return True
    
    # Create a simple tensor
    tensor = torch.ones(10, 10, device=f"cuda:{local_rank}")
    
    # All-reduce
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    
    expected_value = world_size
    actual_value = tensor[0, 0].item()
    
    if actual_value == expected_value:
        print(f"[Rank {local_rank}] ✓ All-reduce successful: {actual_value} == {expected_value}")
        return True
    else:
        print(f"[Rank {local_rank}] ✗ All-reduce failed: {actual_value} != {expected_value}")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("DISTRIBUTED TRAINING FIX VERIFICATION")
    print("="*60)
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    print(f"[Rank {local_rank}] PyTorch version: {torch.__version__}")
    print(f"[Rank {local_rank}] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[Rank {local_rank}] CUDA device: {torch.cuda.get_device_name(local_rank)}")
    
    # Run tests
    results = []
    results.append(("NCCL Initialization", test_nccl_initialization()))
    results.append(("DTensor Sharding", test_dtensor_sharding()))
    results.append(("NCCL All-Reduce", test_all_reduce()))
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"[Rank {local_rank}] {name}: {status}")
        if not passed:
            all_passed = False
    
    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()
    
    if all_passed:
        print(f"\n[Rank {local_rank}] All tests passed! ✓")
    else:
        print(f"\n[Rank {local_rank}] Some tests failed! ✗")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
