"""Fixed hybrid Data Parallel + Model Parallel test.

This version properly handles GPU assignment in multi-process setup.
"""

import os
# Must be set before any other imports
os.environ["KERAS_BACKEND"] = "torch"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.distributed as dist
import numpy as np


def run_hybrid_dp_mp_test():
    """Run the hybrid DP+MP test with proper GPU assignment."""
    
    # Get process info from environment
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # CRITICAL: Each process must use a different GPU
    # In Kaggle, each process gets its own GPU, but torchrun doesn't always
    # set LOCAL_RANK correctly. We need to assign based on process rank.
    if torch.cuda.is_available():
        # Get number of available GPUs
        num_gpus = torch.cuda.device_count()
        
        # Assign GPU based on local_rank (mod to handle more processes than GPUs)
        gpu_id = local_rank % num_gpus
        torch.cuda.set_device(gpu_id)
    
    print(f"\n{'='*70}")
    print(f"TEST: HYBRID DATA PARALLEL + MODEL PARALLEL")
    print(f"{'='*70}")
    print(f"Local rank: {local_rank}")
    print(f"World size: {world_size}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.current_device()}")
        print(f"GPU name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    
    if world_size < 2:
        print("Need at least 2 processes for this test")
        return
    
    # Import Keras distribution classes
    from keras.src.distribution import DeviceMesh, LayoutMap, ModelParallel, initialize
    from torch.distributed._tensor import DTensor
    
    # Initialize distributed backend
    initialize()
    
    # Create DeviceMesh with explicit device assignment
    # Each process creates a mesh that includes all GPUs, but uses different ones
    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    
    mesh = DeviceMesh(
        shape=(world_size,),
        axis_names=["model"],
        devices=devices
    )
    
    print(f"\nDeviceMesh created: shape={mesh.shape}")
    print(f"Devices: {devices}")
    
    # Create LayoutMap for model parallelism with correct patterns
    layout_map = LayoutMap(mesh)
    
    # Use patterns that match keras_hub's path naming (forward slashes)
    layout_map[".*token_embedding.*embeddings"] = ("model",)
    layout_map[".*position_embedding.*embeddings"] = ("model",)
    layout_map[".*segment_embedding.*embeddings"] = ("model",)
    layout_map[".*attention.*query.*kernel"] = ("model",)
    layout_map[".*attention.*key.*kernel"] = ("model",)
    layout_map[".*attention.*value.*kernel"] = ("model",)
    layout_map[".*attention.*output.*kernel"] = ("model",)
    layout_map[".*feedforward.*intermediate.*kernel"] = ("model",)
    layout_map[".*feedforward.*output.*kernel"] = ("model",)
    layout_map[".*pooled_dense.*kernel"] = ("model",)
    layout_map[".*logits.*kernel"] = ("model",)
    
    print("LayoutMap patterns configured")
    
    # Create ModelParallel strategy
    strategy = ModelParallel(
        layout_map=layout_map,
        batch_dim_name="data",
        auto_shard_dataset=False
    )
    
    print("ModelParallel strategy created")
    
    # Build model within strategy scope
    import keras_hub
    import keras
    
    with strategy.scope():
        print(f"\n[Rank {local_rank}] Loading BERT-tiny model...")
        model = keras_hub.models.BertTextClassifier.from_preset(
            "bert_tiny_en_uncased",
            num_classes=2
        )
        print(f"[Rank {local_rank}] ✓ Model loaded")
        
        # Count parameters
        total_params = sum(
            np.prod(w.shape) 
            for layer in model.layers 
            for w in (layer.weights if hasattr(layer, 'weights') else [])
        )
        print(f"[Rank {local_rank}] Total parameters: {total_params:,}")
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=5e-5),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    # Verify sharding
    print(f"\n{'='*70}")
    print(f"TEST: MODEL VERIFICATION")
    print(f"{'='*70}")
    
    sharded_count = 0
    replicated_count = 0
    
    for v in model.variables:
        # Check if the variable is a DTensor
        weight_tensor = None
        
        if isinstance(v, DTensor):
            weight_tensor = v
        elif hasattr(v, 'data') and isinstance(v.data, DTensor):
            weight_tensor = v.data
        elif hasattr(v, '_torch') and isinstance(getattr(v, '_torch', None), DTensor):
            weight_tensor = getattr(v, '_torch')
        
        if weight_tensor is not None:
            local_shape = tuple(weight_tensor.to_local().shape)
            global_shape = tuple(v.shape)
            placements = weight_tensor.placements
            
            if local_shape != global_shape:
                print(f"[Rank {local_rank}] ✓ SHARDED: {v.path}")
                print(f"    Global: {global_shape} -> Local: {local_shape}")
                sharded_count += 1
            else:
                print(f"[Rank {local_rank}]   Replicated: {v.path}")
                replicated_count += 1
    
    print(f"\n[Rank {local_rank}] Sharding Summary:")
    print(f"  Sharded weights: {sharded_count}")
    print(f"  Replicated weights: {replicated_count}")
    
    # Forward pass test
    print(f"\n{'='*70}")
    print(f"TEST: FORWARD PASS")
    print(f"{'='*70}")
    
    batch_size = 2
    texts = ["This is a sample text for testing"] * batch_size
    
    try:
        token_ids_raw = model.preprocessor(texts)
        
        def _to_cpu_numpy(x):
            if isinstance(x, torch.Tensor):
                if x.is_cuda:
                    x = x.cpu()
                return x.numpy()
            return np.array(x)
        
        if isinstance(token_ids_raw, dict):
            token_ids = {
                "token_ids": torch.as_tensor(_to_cpu_numpy(token_ids_raw["token_ids"])).cuda(),
                "padding_mask": torch.as_tensor(_to_cpu_numpy(token_ids_raw["padding_mask"])).cuda(),
                "segment_ids": torch.as_tensor(_to_cpu_numpy(token_ids_raw.get("segment_ids", np.zeros_like(_to_cpu_numpy(token_ids_raw["token_ids"]))))).cuda()
            }
        else:
            token_ids_0 = _to_cpu_numpy(token_ids_raw[0])
            token_ids_1 = _to_cpu_numpy(token_ids_raw[1])
            segment_arr = _to_cpu_numpy(token_ids_raw[2]) if len(token_ids_raw) > 2 else np.zeros_like(token_ids_0)
            token_ids = {
                "token_ids": torch.as_tensor(token_ids_0).cuda(),
                "padding_mask": torch.as_tensor(token_ids_1).cuda(),
                "segment_ids": torch.as_tensor(segment_arr).cuda()
            }
        
        print(f"[Rank {local_rank}] Input shapes: token_ids={token_ids['token_ids'].shape}")
        
        outputs = model(token_ids, training=False)
        
        print(f"[Rank {local_rank}] ✓ Forward pass successful!")
        print(f"Output shape: {outputs.shape}")
        
    except Exception as e:
        print(f"[Rank {local_rank}] ✗ Forward pass failed: {e}")
    
    # Training test
    print(f"\n{'='*70}")
    print(f"TEST: TRAINING")
    print(f"{'='*70}")
    
    labels = np.array([0, 1] * batch_size)
    
    try:
        model.fit(
            texts,
            labels,
            epochs=1,
            batch_size=batch_size,
            verbose=1 if local_rank == 0 else 0
        )
        print(f"[Rank {local_rank}] ✓ Training completed!")
    except Exception as e:
        print(f"[Rank {local_rank}] ✗ Training failed: {e}")
    
    # Cleanup
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    run_hybrid_dp_mp_test()

