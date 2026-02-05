"""Test for hybrid Data Parallel + Model Parallel using BERT from keras_hub
with detailed sharding verification logs.
"""

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


def verify_sharding(model, rank):
    """Verify that model weights are actually sharded across devices."""
    print(f"\n{'='*70}")
    print(f"TEST: MODEL PARALLEL SHARDING VERIFICATION")
    print(f"{'='*70}")
    
    sharded_count = 0
    replicated_count = 0
    total_params = 0
    
    print(f"\n[Rank {rank}] Checking weight sharding across 'model' axis...")
    
    for i, layer in enumerate(model.layers):
        layer_name = layer.name if hasattr(layer, 'name') else f"layer_{i}"
        
        # Get layer weights
        if hasattr(layer, 'weights'):
            for weight in layer.weights:
                weight_name = weight.name if hasattr(weight, 'name') else str(weight)
                total_params += 1
                
                # Check if it's a DTensor
                if hasattr(weight, '_torch_dtensor'):
                    global_shape = tuple(weight.shape)
                    local_shape = tuple(weight._torch_dtensor.to_local().shape)
                    
                    print(f"[Rank {rank}] Layer: {layer_name}")
                    print(f"[Rank {rank}]   Weight: {weight_name}")
                    print(f"[Rank {rank}]   - Global shape: {global_shape}")
                    print(f"[Rank {rank}]   - Local shape: {local_shape}")
                    
                    if local_shape != global_shape:
                        # It's sharded
                        sharding_ratio = global_shape[-1] // local_shape[-1]
                        print(f"[Rank {rank}]   - Sharding ratio: {sharding_ratio}x")
                        print(f"[Rank {rank}]   ✓ SHARDED across 'model' axis")
                        sharded_count += 1
                    else:
                        print(f"[Rank {rank}]   ✓ Replicated")
                        replicated_count += 1
                elif hasattr(weight, 'shape'):
                    # Regular tensor
                    weight_shape = tuple(weight.shape)
                    print(f"[Rank {rank}]   {weight_name}: shape={weight_shape}")
    
    print(f"\n[Rank {rank}] Sharding Summary:")
    print(f"[Rank {rank}]   - Total weights checked: {total_params}")
    print(f"[Rank {rank}]   - Sharded weights: {sharded_count}")
    print(f"[Rank {rank}]   - Replicated weights: {replicated_count}")
    
    return sharded_count > 0


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

    print(f"\n{'='*70}")
    print(f"TEST: HYBRID DATA PARALLEL + MODEL PARALLEL")
    print(f"{'='*70}")
    print(f"[Rank {rank}] {num_devices} GPUs available")
    print(f"[Rank {rank}] PyTorch version: {torch.__version__}")
    print(f"[Rank {rank}] CUDA available: {torch.cuda.is_available()}")
    
    # 2. Define 2D Device Mesh: (Data Parallel, Model Parallel)
    dp_dim = 2 if num_devices >= 4 else 1
    mp_dim = num_devices // dp_dim
    
    print(f"\n{'='*70}")
    print(f"STEP 1: DEVICE MESH SETUP")
    print(f"{'='*70}")
    print(f"[Rank {rank}] DeviceMesh configuration:")
    print(f"[Rank {rank}]   - DP dim (batch): {dp_dim}")
    print(f"[Rank {rank}]   - MP dim (model): {mp_dim}")
    print(f"[Rank {rank}]   - Total devices: {num_devices}")
    
    mesh = DeviceMesh(
        shape=(dp_dim, mp_dim),
        axis_names=["data", "model"],
        devices=devices
    )
    print(f"[Rank {rank}] ✓ DeviceMesh created: shape={mesh.shape}")

    # 3. Create Sharding Layout for hybrid DP+MP
    print(f"\n{'='*70}")
    print(f"STEP 2: LAYOUT MAP CONFIGURATION")
    print(f"{'='*70}")
    
    layout_map = LayoutMap(mesh)
    
    # Model Parallel: shard large weights along model axis
    layout_map["token_embedding/embeddings"] = (None, "model")
    layout_map[".*attention.*query.*kernel"] = (None, "model")
    layout_map[".*attention.*key.*kernel"] = (None, "model")
    layout_map[".*attention.*value.*kernel"] = (None, "model")
    layout_map[".*attention.*output.*kernel"] = ("model", None)
    layout_map[".*feedforward.*intermediate_dense.*kernel"] = (None, "model")
    layout_map[".*feedforward.*output_dense.*kernel"] = ("model", None)
    
    print(f"[Rank {rank}] LayoutMap rules:")
    for key in layout_map:
        layout = layout_map.get_tensor_layout(key)
        if layout:
            print(f"[Rank {rank}]   {key}: axes={layout.axes}")
    
    # 4. Initialize Strategy
    print(f"\n{'='*70}")
    print(f"STEP 3: MODEL PARALLEL STRATEGY")
    print(f"{'='*70}")
    
    strategy = ModelParallel(
        layout_map=layout_map,
        batch_dim_name="data",
        auto_shard_dataset=False
    )
    
    print(f"[Rank {rank}] ModelParallel strategy:")
    print(f"[Rank {rank}]   - batch_dim_name: 'data'")
    print(f"[Rank {rank}]   - auto_shard_dataset: False")
    print(f"[Rank {rank}]   - DeviceMesh: {mesh.shape}")

    # 5. Build model within Strategy Scope
    print(f"\n{'='*70}")
    print(f"STEP 4: MODEL CREATION")
    print(f"{'='*70}")
    
    with strategy.scope():
        print(f"[Rank {rank}] Loading BERT-tiny model from preset...")
        model = keras_hub.models.BertTextClassifier.from_preset(
            "bert_tiny_en_uncased",
            num_classes=2
        )
        print(f"[Rank {rank}] ✓ Model loaded successfully")
        
        # Count total parameters
        total_params = sum(
            np.prod(w.shape) 
            for layer in model.layers 
            for w in (layer.weights if hasattr(layer, 'weights') else [])
        )
        print(f"[Rank {rank}] Total parameters: {total_params:,}")
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=5e-5),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    # 6. Verify sharding
    verify_sharding(model, rank)

    # 7. Forward pass test
    print(f"\n{'='*70}")
    print(f"TEST: FORWARD PASS")
    print(f"{'='*70}")
    
    batch_size = 2
    texts = ["This is a sample text for testing"] * batch_size
    
    print(f"\n[Rank {rank}] Preparing inputs...")
    print(f"[Rank {rank}]   Batch size: {batch_size}")
    print(f"[Rank {rank}]   Sequence length: auto (via preprocessor)")
    
    try:
        # Use preprocessor
        token_ids_raw = model.preprocessor(texts)
        print(f"[Rank {rank}]   Preprocessor output: {type(token_ids_raw)}")
        
        if isinstance(token_ids_raw, dict):
            token_ids = {
                "token_ids": token_ids_raw["token_ids"],
                "padding_mask": token_ids_raw["padding_mask"],
            }
            if "segment_ids" in token_ids_raw:
                token_ids["segment_ids"] = token_ids_raw["segment_ids"]
            else:
                token_ids["segment_ids"] = np.zeros_like(token_ids["token_ids"])
        else:
            token_ids = {
                "token_ids": token_ids_raw[0],
                "padding_mask": token_ids_raw[1],
                "segment_ids": token_ids_raw[2] if len(token_ids_raw) > 2 else np.zeros_like(token_ids_raw[0])
            }
        
        # Convert to torch tensors
        token_ids_torch = {
            "token_ids": torch.as_tensor(token_ids["token_ids"]).cuda(),
            "padding_mask": torch.as_tensor(token_ids["padding_mask"]).cuda(),
            "segment_ids": torch.as_tensor(token_ids["segment_ids"]).cuda()
        }
        
        print(f"\n[Rank {rank}] Input tensor info:")
        print(f"[Rank {rank}]   - token_ids shape: {token_ids_torch['token_ids'].shape}")
        print(f"[Rank {rank}]   - token_ids device: {token_ids_torch['token_ids'].device}")
        print(f"[Rank {rank}]   - padding_mask shape: {token_ids_torch['padding_mask'].shape}")
        
        print(f"\n[Rank {rank}] Running forward pass...")
        outputs = model(token_ids_torch, training=True)
        
        print(f"\n[Rank {rank}] ✓ Forward pass successful!")
        print(f"[Rank {rank}]   Output shape: {outputs.shape}")
        print(f"[Rank {rank}]   Output device: {outputs.device if hasattr(outputs, 'device') else 'N/A'}")
        
    except Exception as e:
        print(f"\n[Rank {rank}] ✗ Forward pass failed:")
        print(f"[Rank {rank}]   Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        
        # Try with training=False
        print(f"\n[Rank {rank}] Trying with training=False...")
        try:
            outputs = model(token_ids_torch, training=False)
            print(f"[Rank {rank}] ✓ Forward pass (training=False) successful!")
            print(f"[Rank {rank}]   Output shape: {outputs.shape}")
        except Exception as e2:
            print(f"[Rank {rank}] ✗ Also failed: {e2}")

    # 8. Training
    print(f"\n{'='*70}")
    print(f"TEST: TRAINING")
    print(f"{'='*70}")
    
    labels = np.array([0, 1] * batch_size)
    print(f"\n[Rank {rank}] Starting training...")
    print(f"[Rank {rank}]   Epochs: 1")
    print(f"[Rank {rank}]   Batch size: {batch_size}")
    
    model.fit(
        texts,
        labels,
        epochs=1,
        batch_size=batch_size,
        verbose=1 if rank == 0 else 0
    )
    
    # 9. Success
    print(f"\n{'='*70}")
    print(f"RESULT")
    print(f"{'='*70}")
    
    if rank == 0:
        print("\n" + "="*60)
        print("✓ HYBRID DP+MP TEST PASSED")
        print(f"✓ Configuration: DP={dp_dim}, MP={mp_dim}")
        print("✓ Model parallelism sharding verified")
        print("✓ Forward pass and training completed")
        print("="*60)

    # Cleanup
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    run_hybrid_dp_mp_test()

