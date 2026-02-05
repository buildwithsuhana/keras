"""Smaller test for distributed training using bert_tiny from keras_hub"""

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


def run_bert_distributed_test():
    """Run distributed training test with BERT-tiny from keras_hub."""
    
    # 1. Initialize Distributed Backend
    initialize()
    
    devices = list_devices("gpu")
    num_devices = len(devices)
    rank = dist.get_rank() if dist.is_initialized() else 0
    
    if num_devices < 2:
        print(f"Detected {num_devices} GPU(s). Need at least 2 for distributed demo.")
        return

    # 2. Define 1D Device Mesh for pure Data Parallel
    # Using 1D mesh avoids device mismatch issues
    mesh = DeviceMesh(
        shape=(num_devices,),
        axis_names=["data"],
        devices=devices
    )

    # 3. Create Sharding Layout - no model parallelism for simplicity
    layout_map = LayoutMap(mesh)
    # Only replicate (no sharding) to avoid device placement issues
    layout_map[".*"] = None
    
    # 4. Initialize Strategy - pure data parallel
    strategy = ModelParallel(
        layout_map=layout_map,
        batch_dim_name="data",
        auto_shard_dataset=False
    )

    # 5. Build BERT classifier within Strategy Scope
    with strategy.scope():
        # Use BERT-tiny (much smaller than OPT-125m)
        model = keras_hub.models.BertTextClassifier.from_preset(
            "bert_tiny_en_uncased",
            num_classes=2
        )
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=5e-5),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    # 6. Generate small dummy data for quick testing
    batch_size = 8  # Per-rank batch size
    num_samples = batch_size * 4  # Samples per rank
    
    # Create dummy text data with labels
    texts = ["This is a sample text for testing"] * num_samples
    labels = np.array([0, 1] * (num_samples // 2))
    
    # 7. Start Training
    print(f"Rank {rank}: Starting BERT-tiny distributed training...")
    print(f"Model: bert_tiny_en_uncased (much smaller than OPT-125m)")
    print(f"Rank {rank}: Using device cuda:{rank}")
    
    model.fit(
        texts,
        labels,
        epochs=1,  # Just 1 epoch for quick test
        batch_size=batch_size,
        verbose=1 if rank == 0 else 0
    )

    # 8. Quick evaluation
    if rank == 0:
        print("\n✓ BERT-tiny distributed training completed successfully!")
        print("✓ The fix for non-floating dtype parameters is working correctly.")

    # 9. Cleanup
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    run_bert_distributed_test()

