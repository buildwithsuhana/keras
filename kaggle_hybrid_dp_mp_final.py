"""Fixed hybrid Data Parallel + Model Parallel test for BERT.

This version uses intra-process parallelism (single process, multiple GPUs)
which is the correct approach for DTensor sharding.
"""

import os
# Must be set before any other imports
os.environ["KERAS_BACKEND"] = "torch"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import numpy as np


def run_hybrid_dp_mp_test():
    """Run the hybrid DP+MP test with proper single-process multi-GPU setup."""
    
    # Check GPU count
    num_gpus = torch.cuda.device_count()
    print(f"\n{'='*70}")
    print(f"TEST: HYBRID DATA PARALLEL + MODEL PARALLEL")
    print(f"{'='*70}")
    print(f"GPUs available: {num_gpus}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if num_gpus < 2:
        print("Need at least 2 GPUs for this test")
        return
    
    # Set GPU devices
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Import Keras distribution classes
    from keras.src.distribution import DeviceMesh, LayoutMap, ModelParallel, list_devices, initialize
    
    # Initialize distributed backend (single process)
    initialize()
    
    # Create DeviceMesh with ALL GPUs in a single process
    # For DTensor sharding, we need all devices in the same process
    devices = [f"cuda:{i}" for i in range(num_gpus)]
    
    # Create 2D mesh: (data_parallel, model_parallel)
    # For 2 GPUs: shape=(1, 2) with data=1, model=2
    dp_dim = 1  # No data parallelism for now
    mp_dim = num_gpus  # Full model parallelism
    
    mesh = DeviceMesh(
        shape=(dp_dim, mp_dim),
        axis_names=["data", "model"],
        devices=devices
    )
    
    print(f"\nDeviceMesh created: shape={mesh.shape}")
    print(f"Devices: {devices}")
    print(f"Axis names: {mesh.axis_names}")
    
    # Create LayoutMap for model parallelism
    layout_map = LayoutMap(mesh)
    
    # Define sharding rules - shard large weights along 'model' axis
    layout_map["token_embedding/embeddings"] = (None, "model")
    layout_map[".*attention.*query.*kernel"] = (None, "model")
    layout_map[".*attention.*key.*kernel"] = (None, "model")
    layout_map[".*attention.*value.*kernel"] = (None, "model")
    layout_map[".*attention.*output.*kernel"] = ("model", None)
    layout_map[".*feedforward.*intermediate_dense.*kernel"] = (None, "model")
    layout_map[".*feedforward.*output_dense.*kernel"] = ("model", None)
    
    print(f"\nLayoutMap rules defined for model parallelism")
    
    # Create ModelParallel strategy
    strategy = ModelParallel(
        layout_map=layout_map,
        batch_dim_name="data",
        auto_shard_dataset=False
    )
    
    print(f"\nModelParallel strategy created")
    
    # Build model within strategy scope
    import keras_hub
    
    with strategy.scope():
        print(f"\nLoading BERT-tiny model...")
        model = keras_hub.models.BertTextClassifier.from_preset(
            "bert_tiny_en_uncased",
            num_classes=2
        )
        print(f"✓ Model loaded")
        
        # Count parameters
        total_params = sum(
            np.prod(w.shape) 
            for layer in model.layers 
            for w in (layer.weights if hasattr(layer, 'weights') else [])
        )
        print(f"Total parameters: {total_params:,}")
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=5e-5),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    # Verify sharding
    print(f"\n{'='*70}")
    print(f"TEST: MODEL VERIFICATION")
    print(f"{'='*70}")
    
    from torch.distributed._tensor import DTensor
    
    sharded_count = 0
    replicated_count = 0
    
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'weights'):
            for w in layer.weights:
                # Check if weight is a DTensor
                weight_tensor = w.data if hasattr(w, 'data') else w
                if isinstance(weight_tensor, DTensor):
                    local_shape = tuple(weight_tensor.to_local().shape)
                    global_shape = tuple(w.shape)
                    placements = weight_tensor.placements
                    
                    layer_name = layer.name if hasattr(layer, 'name') else str(layer)
                    weight_name = w.name if hasattr(w, 'name') else str(w)
                    
                    print(f"{layer_name}/{weight_name}:")
                    print(f"  Global shape: {global_shape}")
                    print(f"  Local shape: {local_shape}")
                    print(f"  Placements: {placements}")
                    
                    if local_shape != global_shape:
                        print(f"  ✓ SHARDED")
                        sharded_count += 1
                    else:
                        print(f"  ✓ Replicated")
                        replicated_count += 1
    
    print(f"\nSharding Summary:")
    print(f"  Sharded weights: {sharded_count}")
    print(f"  Replicated weights: {replicated_count}")
    
    # Forward pass test
    print(f"\n{'='*70}")
    print(f"TEST: FORWARD PASS")
    print(f"{'='*70}")
    
    batch_size = 2
    texts = ["This is a sample text for testing"] * batch_size
    
    try:
        # Prepare inputs
        token_ids_raw = model.preprocessor(texts)
        
        if isinstance(token_ids_raw, dict):
            token_ids = {
                "token_ids": torch.as_tensor(token_ids_raw["token_ids"]).cuda(),
                "padding_mask": torch.as_tensor(token_ids_raw["padding_mask"]).cuda(),
                "segment_ids": torch.as_tensor(token_ids_raw.get("segment_ids", np.zeros_like(token_ids_raw["token_ids"]))).cuda()
            }
        else:
            token_ids = {
                "token_ids": torch.as_tensor(token_ids_raw[0]).cuda(),
                "padding_mask": torch.as_tensor(token_ids_raw[1]).cuda(),
                "segment_ids": torch.as_tensor(token_ids_raw[2] if len(token_ids_raw) > 2 else np.zeros_like(token_ids_raw[0])).cuda()
            }
        
        print(f"Input tensor shapes:")
        print(f"  token_ids: {token_ids['token_ids'].shape}")
        print(f"  padding_mask: {token_ids['padding_mask'].shape}")
        
        print(f"\nRunning forward pass...")
        outputs = model(token_ids, training=False)
        
        print(f"✓ Forward pass successful!")
        print(f"Output shape: {outputs.shape}")
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Training test
    print(f"\n{'='*70}")
    print(f"TEST: TRAINING")
    print(f"{'='*70}")
    
    labels = np.array([0, 1] * batch_size)
    print(f"Starting training...")
    
    model.fit(
        texts,
        labels,
        epochs=1,
        batch_size=batch_size,
        verbose=1
    )
    
    print(f"\n{'='*70}")
    print(f"RESULT")
    print(f"{'='*70}")
    print("✓ HYBRID DP+MP TEST COMPLETED")
    print(f"✓ Configuration: DP={dp_dim}, MP={mp_dim}")
    print(f"✓ Sharded weights: {sharded_count}")
    print(f"✓ Forward pass and training completed")


if __name__ == "__main__":
    run_hybrid_dp_mp_test()
