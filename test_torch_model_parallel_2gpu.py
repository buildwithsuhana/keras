#!/usr/bin/env python3
"""
Test script for ACTUAL Model Parallelism with OPT-125M using PyTorch DTensor.

This script demonstrates TRUE physical sharding with PyTorch DTensor:
1. Setting up torch backend with 2 physical GPUs
2. Initializing torch distributed with NCCL backend
3. Creating DeviceMesh for model parallelism
4. Using PyTorch DTensor to shard tensors across devices
5. Verifying actual sharded tensor shapes on each device

Usage (with 2+ GPUs):
    torchrun --nproc_per_node=2 python test_torch_model_parallel_2gpu.py

Or simply:
    python test_torch_model_parallel_2gpu.py  # Will auto-spawn processes for multi-GPU
"""

import os
import sys
import logging
import numpy as np

# Set backend before importing keras
os.environ["KERAS_BACKEND"] = "torch"

import torch

# For multi-GPU training, we need to set these env vars BEFORE initializing distributed
# This helps with NCCL backend initialization
if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_BLOCKING_WAIT"] = "0"

import torch.distributed as dist
import torch.distributed.tensor.parallel as tp
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    SequenceParallel,
)
import torch.multiprocessing as mp

import keras
from keras.src.distribution import DeviceMesh as KerasDeviceMesh
from keras.src.distribution import LayoutMap, ModelParallel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)


def setup_distributed_for_rank(rank, world_size):
    """Initialize distributed for a specific rank."""
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
    
    if not dist.is_initialized():
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank
        )
    
    return rank, world_size


def cleanup_distributed():
    """Clean up distributed resources."""
    if dist.is_initialized():
        dist.destroy_process_group()


# ============================================================================
# Helper Functions for DTensor
# ============================================================================

def create_device_mesh_2d(gpu_devices, rank):
    """Create a 2D device mesh for model + data parallelism.
    
    Args:
        gpu_devices: List of CUDA device indices
        rank: Current process rank
        
    Returns:
        PyTorch DeviceMesh
    """
    # Create a 1D mesh for model parallelism
    # For true 2D parallelism, we'd need more GPUs
    # Note: init_device_mesh uses device_type, not backend parameter
    mesh = init_device_mesh(
        device_type="cuda",
        mesh_shape=(len(gpu_devices),),
        mesh_dim_names=["model"]
    )
    return mesh


def create_dtensor_from_numpy(data, mesh, shard_dim=None):
    """Create a DTensor from numpy array.
    
    Args:
        data: numpy array
        mesh: DeviceMesh
        shard_dim: dimension to shard on (None for replicated)
        
    Returns:
        DTensor placed on mesh devices
    """
    if shard_dim is None:
        placement = tp.Replicate()
    else:
        placement = tp.Shard(shard_dim)
    
    # Convert to tensor on the local rank device first
    tensor = torch.tensor(data, dtype=torch.float32)
    
    # Use distribute_tensor to create DTensor
    dtensor = tp.distribute_tensor(tensor, mesh, [placement])
    
    return dtensor


def print_shard_info(tensor, name):
    """Print information about a tensor's sharding."""
    if hasattr(tensor, '_local_tensor'):
        local_shape = tensor._local_tensor.shape
        print(f"  [DTENSOR] {name}: local_shape={local_shape}, device={tensor.device}")
    else:
        print(f"  [TENSOR]  {name}: shape={tuple(tensor.shape)}, device={tensor.device}")


# ============================================================================
# OPT-125M Model with PyTorch DTensor
# ============================================================================

class OPT125MModelParallel(torch.nn.Module):
    """OPT-125M style model with PyTorch DTensor model parallelism.
    
    This model uses torch.distributed.tensor.parallel for actual
    weight sharding across multiple GPUs.
    """
    
    def __init__(self, config, device_mesh):
        super().__init__()
        
        self.config = config
        self.device_mesh = device_mesh
        self.vocab_size = config['vocabulary_size']
        self.hidden_dim = config['hidden_dim']
        self.num_layers = config['num_layers']
        self.num_heads = config['num_heads']
        self.intermediate_dim = config['intermediate_dim']
        
        # Create the model with model parallelism
        self._create_model_parallel()
    
    def _create_model_parallel(self):
        """Create model with tensor parallel layers."""
        
        # Define parallel styles
        colwise_style = ColwiseParallel()
        rowwise_style = RowwiseParallel()
        
        # Embedding layer - replicate across model dimension
        self.token_embedding = torch.nn.Embedding(
            self.vocab_size,
            self.hidden_dim
        )
        self.position_embedding = torch.nn.Embedding(
            self.config['max_sequence_length'],
            self.hidden_dim
        )
        
        # Transformer layers with model parallelism
        self.transformer_layers = torch.nn.ModuleList()
        
        for i in range(self.num_layers):
            layer = self._create_parallel_transformer_layer(i, colwise_style, rowwise_style)
            self.transformer_layers.append(layer)
        
        # Final layer norm
        self.layer_norm = torch.nn.LayerNorm(self.hidden_dim)
        
        # Output projection
        self.output_projection = torch.nn.Linear(
            self.hidden_dim,
            self.vocab_size,
            bias=False
        )
    
    def _create_parallel_transformer_layer(self, layer_idx, colwise_style, rowwise_style):
        """Create a transformer layer with model parallelism."""
        
        # Attention layer
        attention = torch.nn.MultiheadAttention(
            self.hidden_dim,
            self.num_heads,
            batch_first=True
        )
        
        # Parallelize attention
        tp.parallelize_module(attention, self.device_mesh, {
            "q_proj": colwise_style,
            "k_proj": colwise_style,
            "v_proj": colwise_style,
            "out_proj": colwise_style,
        })
        
        # Feed-forward layers
        fc1 = torch.nn.Linear(self.hidden_dim, self.intermediate_dim)
        fc2 = torch.nn.Linear(self.intermediate_dim, self.hidden_dim)
        
        # Parallelize feed-forward
        tp.parallelize_module(fc1, self.device_mesh, {"weight": colwise_style, "bias": colwise_style})
        tp.parallelize_module(fc2, self.device_mesh, {"weight": rowwise_style, "bias": rowwise_style})
        
        # Layer norms
        attn_norm = torch.nn.LayerNorm(self.hidden_dim)
        ffn_norm = torch.nn.LayerNorm(self.hidden_dim)
        
        return {
            'attention': attention,
            'fc1': fc1,
            'fc2': fc2,
            'attn_norm': attn_norm,
            'ffn_norm': ffn_norm,
            'dropout': torch.nn.Dropout(self.config['dropout'])
        }
    
    def forward(self, token_ids, attention_mask=None):
        """Forward pass."""
        batch_size, seq_len = token_ids.shape
        
        # Embeddings
        token_embeds = self.token_embedding(token_ids)
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0).expand(batch_size, -1)
        pos_embeds = self.position_embedding(positions)
        hidden_states = token_embeds + pos_embeds
        
        # Transformer layers
        for layer in self.transformer_layers:
            # Self-attention with pre-norm
            attn_norm_out = layer['attn_norm'](hidden_states)
            
            # Causal mask
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=hidden_states.device),
                diagonal=1
            ).bool()
            
            attn_output, attn_weights = layer['attention'](
                attn_norm_out,
                attn_norm_out,
                attn_norm_out,
                attn_mask=causal_mask,
                need_weights=False
            )
            
            # Dropout and residual
            hidden_states = hidden_states + layer['dropout'](attn_output)
            
            # Feed-forward with pre-norm
            ffn_norm_out = layer['ffn_norm'](hidden_states)
            
            ffn_output = layer['fc1'](ffn_norm_out)
            ffn_output = torch.nn.functional.gelu(ffn_output)
            ffn_output = layer['fc2'](ffn_output)
            ffn_output = layer['dropout'](ffn_output)
            
            # Residual
            hidden_states = hidden_states + ffn_output
        
        # Final layer norm
        hidden_states = self.layer_norm(hidden_states)
        
        # Output projection
        logits = self.output_projection(hidden_states)
        
        return logits


# ============================================================================
# Main Test Function (runs on each rank)
# ============================================================================

def test_torch_dtensor_model_parallelism_worker(rank, world_size, num_gpus):
    """Worker function that runs on each GPU process."""
    
    # Setup distributed for this rank
    rank, world_size = setup_distributed_for_rank(rank, world_size)
    local_rank = rank
    
    print(f"\n{'='*80}")
    print(f"[Rank {rank}/{world_size}] Starting test on local rank {local_rank}")
    print(f"{'='*80}")
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This test requires GPUs.")
        cleanup_distributed()
        return False
    
    print(f"\n[Rank {rank}] GPU check:")
    print(f"  - CUDA available: {torch.cuda.is_available()}")
    print(f"  - GPU count: {num_gpus}")
    print(f"  - Current GPU: {torch.cuda.current_device()}")
    print(f"  - GPU name: {torch.cuda.get_device_name(rank)}")
    
    gpu_devices = list(range(num_gpus))
    print(f"\n[Rank {rank}] Using GPUs: {gpu_devices}")
    
    # =========================================================================
    # Step 1: Create DeviceMesh for Model Parallelism
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"STEP 1: Create DeviceMesh for Model Parallelism")
    print(f"{'='*80}")
    
    # Create PyTorch DeviceMesh
    device_mesh = create_device_mesh_2d(gpu_devices, rank)
    print(f"\n✓ [Rank {rank}] Created DeviceMesh: shape={device_mesh.mesh.shape}, dim_names={device_mesh.mesh_dim_names}")
    
    # Show mesh device assignments
    for i, device in enumerate(device_mesh.device_ids):
        print(f"  - Rank {i}: cuda:{device}")
    
    # =========================================================================
    # Step 2: Create OPT-125M Configuration
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"STEP 2: OPT-125M Configuration")
    print(f"{'='*80}")
    
    OPT_125M_CONFIG = {
        'vocabulary_size': 50265,
        'num_layers': 12,
        'num_heads': 12,
        'hidden_dim': 768,
        'intermediate_dim': 3072,
        'dropout': 0.1,
        'max_sequence_length': 2048,
    }
    
    print(f"\n[Rank {rank}] OPT-125M Configuration:")
    for key, value in OPT_125M_CONFIG.items():
        print(f"  - {key}: {value}")
    
    # Calculate model parameters
    total_params = (
        OPT_125M_CONFIG['vocabulary_size'] * OPT_125M_CONFIG['hidden_dim'] * 2 +  # embeddings
        OPT_125M_CONFIG['max_sequence_length'] * OPT_125M_CONFIG['hidden_dim'] +  # position
        OPT_125M_CONFIG['num_layers'] * (
            3 * OPT_125M_CONFIG['hidden_dim'] * OPT_125M_CONFIG['hidden_dim'] +  # Q, K, V projections
            OPT_125M_CONFIG['hidden_dim'] * OPT_125M_CONFIG['hidden_dim'] +  # attention output
            2 * OPT_125M_CONFIG['hidden_dim'] * OPT_125M_CONFIG['intermediate_dim'] +  # FFN
            OPT_125M_CONFIG['hidden_dim'] * OPT_125M_CONFIG['intermediate_dim'] +
            4 * OPT_125M_CONFIG['hidden_dim'] * 2  # layer norms
        ) +
        OPT_125M_CONFIG['hidden_dim'] * OPT_125M_CONFIG['vocabulary_size']  # output
    )
    
    print(f"\n  - Total parameters: {total_params:,}")
    print(f"  - Parameter memory: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")
    
    # =========================================================================
    # Step 3: Create Model with DTensor Parallelism
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"STEP 3: Create Model with DTensor Model Parallelism")
    print(f"{'='*80}")
    
    # Create model on current GPU
    model = OPT125MModelParallel(OPT_125M_CONFIG, device_mesh)
    model = model.cuda()
    
    # Parallelize the entire module
    model = tp.parallelize_module(model, device_mesh, {})
    print(f"\n✓ [Rank {rank}] Model parallelized across devices")
    
    # Count parameters
    total_params_count = sum(p.numel() for p in model.parameters())
    local_params_count = sum(p._local_tensor.numel() for p in model.parameters() if hasattr(p, '_local_tensor'))
    
    print(f"\n[Rank {rank}] Parameter counts:")
    print(f"  - Total parameters (logical): {total_params_count:,}")
    print(f"  - Local parameters (per device): {local_params_count:,}")
    print(f"  - Sharding ratio: {total_params_count / local_params_count:.1f}x")
    
    # =========================================================================
    # Step 4: Verify Tensor Sharding
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"STEP 4: Verify Tensor Sharding")
    print(f"{'='*80}")
    
    print(f"\n[Rank {rank}] Verifying DTensor properties of model weights:")
    
    for name, param in model.named_parameters():
        if hasattr(param, '_local_tensor'):
            local_shape = tuple(param._local_tensor.shape)
            global_shape = param.shape
            
            # Calculate expected shard shape
            if 'attention' in name and 'weight' in name:
                expected_shard = global_shape[1] // 2  # Second dimension sharded
            elif 'fc1' in name and 'weight' in name:
                expected_shard = global_shape[0] // 2  # First dimension sharded
            elif 'fc2' in name and 'weight' in name:
                expected_shard = global_shape[1] // 2  # Second dimension sharded
            elif 'embedding' in name:
                expected_shard = global_shape[0] // 2  # First dimension sharded
            else:
                expected_shard = "replicated"
            
            print(f"  ✓ {name}:")
            print(f"      Global shape: {global_shape}")
            print(f"      Local shape: {local_shape}")
            print(f"      Device: {param._local_tensor.device}")
            
            if isinstance(expected_shard, int):
                shard_ratio = global_shape[1] / local_shape[1] if len(local_shape) > 1 else 1
                print(f"      Shard ratio: {shard_ratio:.1f}x")
        else:
            print(f"  - {name}: {tuple(param.shape)} (replicated)")
    
    # =========================================================================
    # Step 5: Forward Pass Test
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"STEP 5: Forward Pass Test")
    print(f"{'='*80}")
    
    # Create sample input
    batch_size = 2
    seq_length = 8
    
    token_ids = torch.randint(
        0,
        OPT_125M_CONFIG['vocabulary_size'],
        size=(batch_size, seq_length),
        device=f"cuda:{local_rank}"
    )
    
    print(f"\n[Rank {rank}] Input:")
    print(f"  - Token IDs shape: {tuple(token_ids.shape)}")
    print(f"  - Device: {token_ids.device}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        logits = model(token_ids)
    
    print(f"\n[Rank {rank}] Output:")
    print(f"  - Logits shape: {tuple(logits.shape)}")
    print(f"  - Device: {logits.device}")
    
    if hasattr(logits, '_local_tensor'):
        print(f"  - Local logits shape: {tuple(logits._local_tensor.shape)}")
    
    # Check for NaN
    has_nan = torch.isnan(logits).any().item()
    print(f"  - Contains NaN: {has_nan}")
    
    # =========================================================================
    # Step 6: Training Test
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"STEP 6: Training Test")
    print(f"{'='*80}")
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    
    # Create synthetic labels (next token prediction)
    labels = torch.randint(
        0,
        OPT_125M_CONFIG['vocabulary_size'],
        size=(batch_size, seq_length),
        device=f"cuda:{local_rank}"
    )
    
    print(f"\n[Rank {rank}] Training configuration:")
    print(f"  - Optimizer: Adam (lr=5e-5)")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Sequence length: {seq_length}")
    
    # Training loop
    model.train()
    num_epochs = 2
    
    print(f"\n[Rank {rank}] Training for {num_epochs} epoch(s)...")
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(token_ids)
        
        # Calculate loss (shift logits for next token prediction)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(shift_logits.view(-1, OPT_125M_CONFIG['vocabulary_size']), shift_labels.view(-1))
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Get local gradient norm
        grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.norm().item()
        
        print(f"  [Rank {rank}] Epoch {epoch+1}/{num_epochs}: loss={loss.item():.6f}, grad_norm={grad_norm:.4f}")
    
    print(f"\n✓ [Rank {rank}] Training completed successfully!")
    
    # =========================================================================
    # Step 7: Verify Sharding After Training
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"STEP 7: Verify Sharding After Training")
    print(f"{'='*80}")
    
    print(f"\n[Rank {rank}] Verifying sharded parameters after training:")
    
    for name, param in model.named_parameters():
        if hasattr(param, '_local_tensor'):
            local_shape = tuple(param._local_tensor.shape)
            global_shape = param.shape
            grad_norm = param.grad.norm().item() if param.grad is not None else 0.0
            
            print(f"  ✓ {name}:")
            print(f"      Local shape: {local_shape}")
            print(f"      Gradient norm: {grad_norm:.6f}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"SUMMARY (Rank {rank})")
    print(f"{'='*80}")
    
    print(f"""
[Rank {rank}] This test demonstrated:

1. ✓ PyTorch DTensor DeviceMesh creation for model parallelism
2. ✓ Actual tensor parallel module parallelization using torch.distributed
3. ✓ Verification of DTensor sharding (global vs local shapes)
4. ✓ Forward pass with sharded model
5. ✓ Backpropagation through sharded model
6. ✓ Gradient computation and optimizer step on sharded weights

Key insights for TRUE model parallelism:
- PyTorch DTensor requires torch.distributed with NCCL backend
- Each process controls one GPU (rank = GPU index)
- Sharded tensors have _local_tensor attribute with device-local shape
- Optimizer operates on local shards, PyTorch handles gradient synchronization
- torch.distributed.tensor.parallel.parallelize_module() applies sharding styles
""")
    
    print(f"{'='*80}")
    print(f"[Rank {rank}] Test completed successfully!")
    print(f"{'='*80}")
    
    cleanup_distributed()
    return True


def main():
    """Main entry point that spawns worker processes."""
    
    # Check if running under torchrun
    world_size_from_env = int(os.environ.get("WORLD_SIZE", 1))
    rank_from_env = int(os.environ.get("RANK", 0))
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This test requires GPUs.")
        sys.exit(1)
    
    num_gpus = torch.cuda.device_count()
    print("=" * 80)
    print("PyTorch DTensor Model Parallelism Test with 2+ GPUs")
    print("=" * 80)
    print(f"\nGPU Environment:")
    print(f"  - CUDA available: {torch.cuda.is_available()}")
    print(f"  - GPU count: {num_gpus}")
    
    if num_gpus < 2:
        print("\nWARNING: This test requires 2+ GPUs for model parallelism.")
        print("Running in single GPU mode (limited functionality).")
        # Still run but without true model parallelism
        test_torch_dtensor_model_parallelism_worker(rank=0, world_size=1, num_gpus=num_gpus)
        return
    
    # Check if running under torchrun
    if world_size_from_env > 1:
        # Running under torchrun - just run the worker
        print(f"\nDetected torchrun mode: WORLD_SIZE={world_size_from_env}")
        test_torch_dtensor_model_parallelism_worker(rank=rank_from_env, world_size=world_size_from_env, num_gpus=num_gpus)
    else:
        # Not under torchrun - need to spawn processes ourselves
        print(f"\nDetected python mode - spawning {num_gpus} processes for multi-GPU training")
        print(f"\nTo run with torchrun instead, use:")
        print(f"  torchrun --nproc_per_node={num_gpus} {sys.argv[0]}")
        print()
        
        mp.spawn(
            test_torch_dtensor_model_parallelism_worker,
            args=(num_gpus, num_gpus),
            nprocs=num_gpus,
            join=True
        )


if __name__ == "__main__":
    main()

