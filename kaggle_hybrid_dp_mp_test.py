"""Test for hybrid Data Parallel + Model Parallel using BERT from keras_hub"""

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


def run_hybrid_dp_mp_test():
    """Test hybrid Data Parallel + Model Parallel with BERT-tiny."""
    
    # 1. Initialize Distributed Backend
    initialize()
    
    devices = list_devices("gpu")
    num_devices = len(devices)
    rank = dist.get_rank() if dist.is_initialized() else 0
    
    if num_devices < 2:
        print(f"Detected {num_devices} GPU(s). Need at least 2 for hybrid DP+MP demo.")
        return

    print(f"Rank {rank}: {num_devices} GPUs available")
    
    # 2. Define 2D Device Mesh: (Data Parallel, Model Parallel)
    # For 2 GPUs: dp=1, mp=2 (pure MP) OR dp=2, mp=1 (pure DP)
    # For 4 GPUs: dp=2, mp=2 (hybrid)
    dp_dim = 2 if num_devices >= 4 else 1
    mp_dim = num_devices // dp_dim
    
    print(f"Rank {rank}: DP dim={dp_dim}, MP dim={mp_dim}")
    
    mesh = DeviceMesh(
        shape=(dp_dim, mp_dim),
        axis_names=["data", "model"],
        devices=devices
    )

    # 3. Create Sharding Layout for hybrid DP+MP
    layout_map = LayoutMap(mesh)
    
    # Model Parallel: shard large weights along model axis
    layout_map["token_embedding/embeddings"] = (None, "model")
    layout_map[".*attention.*query.*kernel"] = (None, "model")
    layout_map[".*attention.*key.*kernel"] = (None, "model")
    layout_map[".*attention.*value.*kernel"] = (None, "model")
    layout_map[".*attention.*output.*kernel"] = ("model", None)
    layout_map[".*feedforward.*intermediate_dense.*kernel"] = (None, "model")
    layout_map[".*feedforward.*output_dense.*kernel"] = ("model", None)
    
    # 4. Initialize Strategy
    strategy = ModelParallel(
        layout_map=layout_map,
        batch_dim_name="data",
        auto_shard_dataset=False
    )

    # 5. Build model within Strategy Scope
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

    # 6. Generate data - ensure proper device placement
    batch_size = 4  # Per-rank batch
    seq_length = 32
    
    texts = ["This is a sample text for testing"] * (batch_size * 2)
    labels = np.array([0, 1] * (batch_size))
    
    # 7. Manual forward pass to verify device placement
    print(f"Rank {rank}: Running forward pass...")
    
    # Run one forward pass manually to check for device issues
    try:
        # Tokenize - keras_hub tokenizer returns a tuple/list
        token_ids_raw = model.preprocessor.tokenizer(texts)
        print(f"Rank {rank}: Tokenizer output type: {type(token_ids_raw)}")
        
        # Convert to dict - keras_hub returns (token_ids, attention_mask)
        if isinstance(token_ids_raw, (list, tuple)):
            token_ids = {
                "token_ids": token_ids_raw[0],
                "padding_mask": token_ids_raw[1],
                "segment_ids": token_ids_raw[2] if len(token_ids_raw) > 2 else None  # BERT needs segment_ids
            }
        else:
            token_ids = token_ids_raw
            
        # Convert to torch tensors with proper device
        import torch
        token_ids_torch = {
            "token_ids": torch.as_tensor(token_ids["token_ids"]).cuda(),
            "padding_mask": torch.as_tensor(token_ids["padding_mask"]).cuda(),
            "segment_ids": torch.as_tensor(token_ids["segment_ids"]).cuda() if token_ids["segment_ids"] is not None else None
        }
        
        print(f"Rank {rank}: Token IDs shape: {token_ids_torch['token_ids'].shape}")
        print(f"Rank {rank}: Token IDs device: {token_ids_torch['token_ids'].device}")
        
        # Forward pass
        outputs = model(token_ids_torch, training=True)
        
        print(f"Rank {rank}: Forward pass successful!")
        print(f"Rank {rank}: Output shape: {outputs.shape}")
        
    except RuntimeError as e:
        if "device" in str(e).lower():
            print(f"Rank {rank}: Device mismatch detected: {e}")
            # Try with explicit device placement
            print(f"Rank {rank}: Attempting fix with explicit device placement...")
            
            # Move inputs to current device
            current_device = torch.cuda.current_device()
            token_ids_cuda = {
                "token_ids": torch.as_tensor(token_ids["token_ids"]).cuda(current_device),
                "attention_mask": torch.as_tensor(token_ids["attention_mask"]).cuda(current_device)
            }
            
            outputs = model(token_ids_cuda, training=True)
            print(f"Rank {rank}: Forward pass successful with explicit device!")
        else:
            raise

    # 8. Training
    print(f"Rank {rank}: Starting training...")
    
    model.fit(
        texts,
        labels,
        epochs=1,
        batch_size=batch_size,
        verbose=1 if rank == 0 else 0
    )

    # 9. Success
    if rank == 0:
        print("\n" + "="*60)
        print("✓ Hybrid DP+MP training completed successfully!")
        print(f"✓ Configuration: DP={dp_dim}, MP={mp_dim}")
        print("✓ The fix for non-floating dtype parameters is working!")
        print("="*60)

    # Cleanup
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    run_hybrid_dp_mp_test()

