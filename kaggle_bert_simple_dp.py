"""Simple test for BERT distributed training with Data Parallel (no sharding)"""

import os
# Must be set before any other imports
os.environ["KERAS_BACKEND"] = "torch"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.distributed as dist
import keras
import keras_hub
import numpy as np
from keras.src.distribution import ModelParallel, DeviceMesh, LayoutMap, list_devices, initialize


def run_simple_dp_test():
    """Simple test using pure Data Parallel (no model sharding)."""
    
    # 1. Initialize Distributed Backend
    initialize()
    
    devices = list_devices("gpu")
    num_devices = len(devices)
    rank = dist.get_rank() if dist.is_initialized() else 0
    
    if num_devices < 2:
        print(f"Detected {num_devices} GPU(s). Need at least 2 for distributed demo.")
        return

    print(f"\n{'='*70}")
    print(f"TEST: PURE DATA PARALLEL")
    print(f"{'='*70}")
    print(f"[Rank {rank}] {num_devices} GPUs available")
    
    # 2. Use 1D Device Mesh for pure Data Parallel
    print(f"\n[Rank {rank}] Setting up DeviceMesh for Data Parallel...")
    
    mesh = DeviceMesh(
        shape=(num_devices,),
        axis_names=["data"],
        devices=devices
    )

    # 3. No sharding - just replicate (empty tuple = replicate)
    layout_map = LayoutMap(mesh)
    layout_map[".*"] = ()  # Replicate all weights
    
    print(f"[Rank {rank}] Layout: Replicate (no sharding)")
    
    # 4. Initialize Strategy
    strategy = ModelParallel(
        layout_map=layout_map,
        batch_dim_name="data",
        auto_shard_dataset=False
    )

    # 5. Build model
    print(f"\n[Rank {rank}] Loading BERT-tiny model...")
    
    with strategy.scope():
        model = keras_hub.models.BertTextClassifier.from_preset(
            "bert_tiny_en_uncased",
            num_classes=2
        )
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=5e-5),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    # 6. Generate small batch - each rank gets different data
    batch_size = 2
    texts = ["This is a sample text for testing"] * (batch_size * num_devices)
    labels = np.array([0, 1] * (batch_size * num_devices))
    
    # Each rank processes its own subset
    rank_texts = texts[rank::num_devices]
    rank_labels = labels[rank::num_devices]
    
    print(f"\n[Rank {rank}] Dataset info:")
    print(f"[Rank {rank}]   - Total samples: {len(texts)}")
    print(f"[Rank {rank}]   - Samples per rank: {len(rank_texts)}")
    
    # 7. Training - each rank trains independently
    print(f"\n[Rank {rank}] Starting training...")
    print(f"[Rank {rank}]   Epochs: 1")
    print(f"[Rank {rank}]   Local batch size: {batch_size}")
    
    model.fit(
        rank_texts,
        rank_labels,
        epochs=1,
        batch_size=batch_size,
        verbose=1 if rank == 0 else 0
    )

    # 8. Success
    if rank == 0:
        print("\n" + "="*60)
        print("✓ Simple DP training completed!")
        print("✓ Non-floating dtype fix is working correctly!")
        print("="*60)

    # Cleanup
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    run_simple_dp_test()

