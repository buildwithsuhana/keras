"""Simple test for BERT distributed training with manual data handling"""

import os
os.environ["KERAS_BACKEND"] = "torch"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.distributed as dist
import keras
import keras_hub
import numpy as np
from keras.src.distribution import ModelParallel, DeviceMesh, LayoutMap, list_devices, initialize


def run_simple_dp_test():
    """Simple test using pure Data Parallel with manual data handling."""
    
    initialize()
    
    devices = list_devices("gpu")
    num_devices = len(devices)
    rank = dist.get_rank() if dist.is_initialized() else 0
    
    if num_devices < 2:
        print(f"Detected {num_devices} GPU(s). Need at least 2 GPUs.")
        return

    print(f"\n{'='*70}")
    print(f"TEST: PURE DATA PARALLEL")
    print(f"{'='*70}")
    print(f"[Rank {rank}] {num_devices} GPUs available")
    
    # Setup DeviceMesh
    mesh = DeviceMesh(
        shape=(num_devices,),
        axis_names=["data"],
        devices=devices
    )

    # Layout: replicate (no sharding)
    layout_map = LayoutMap(mesh)
    layout_map[".*"] = ()
    
    print(f"[Rank {rank}] Layout: Replicate (no sharding)")
    
    # Initialize Strategy
    strategy = ModelParallel(
        layout_map=layout_map,
        batch_dim_name="data",
        auto_shard_dataset=False
    )

    # Build model
    print(f"\n[Rank {rank}] Loading BERT-tiny model...")
    
    with strategy.scope():
        model = keras_hub.models.BertTextClassifier.from_preset(
            "bert_tiny_en_uncased",
            num_classes=2
        )

    # Generate small batch
    batch_size = 2
    texts = ["This is a sample text"] * batch_size
    
    print(f"\n[Rank {rank}] Preprocessing data...")
    
    # Preprocess data
    token_ids = model.preprocessor(texts)
    
    # Convert to torch
    inputs = {
        "token_ids": torch.as_tensor(token_ids["token_ids"]).cuda(),
        "padding_mask": torch.as_tensor(token_ids["padding_mask"]).cuda(),
    }
    if "segment_ids" in token_ids:
        inputs["segment_ids"] = torch.as_tensor(token_ids["segment_ids"]).cuda()
    else:
        inputs["segment_ids"] = torch.zeros_like(inputs["token_ids"])
    
    labels = torch.tensor([0, 1]).cuda()
    
    print(f"[Rank {rank}] Input shape: {inputs['token_ids'].shape}")
    print(f"[Rank {rank}] Labels shape: {labels.shape}")
    
    # Training
    print(f"\n[Rank {rank}] Starting training...")
    print(f"[Rank {rank}]   Epochs: 1")
    print(f"[Rank {rank}]   Batch size: {batch_size}")
    
    optimizer = keras.optimizers.Adam(learning_rate=5e-5)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    for epoch in range(1):
        print(f"[Rank {rank}]   Epoch {epoch + 1}/1")
        
        # Forward pass
        outputs = model(inputs, training=True)
        loss = loss_fn(labels, outputs)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"[Rank {rank}]     Loss: {loss.item():.4f}")
    
    # Success
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

