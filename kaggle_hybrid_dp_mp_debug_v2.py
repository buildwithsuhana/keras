"""Debug hybrid Data Parallel + Model Parallel test.

This version prints actual variable paths to fix LayoutMap patterns.
"""

import os
# Must be set before any other imports
os.environ["KERAS_BACKEND"] = "torch"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import numpy as np


def run_hybrid_dp_mp_test():
    """Run the hybrid DP+MP test to debug variable paths."""
    
    # Check GPU count and process info
    num_gpus = torch.cuda.device_count()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    print(f"\n{'='*70}")
    print(f"TEST: HYBRID DATA PARALLEL + MODEL PARALLEL")
    print(f"{'='*70}")
    print(f"GPUs available: {num_gpus}")
    print(f"World size: {world_size}")
    print(f"Local rank: {local_rank}")
    
    if world_size < 2 and num_gpus < 2:
        print("Need at least 2 GPUs for this test")
        return
    
    # Set GPU device
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    
    # Import Keras distribution classes
    from keras.src.distribution import DeviceMesh, LayoutMap, ModelParallel, initialize
    from torch.distributed._tensor import DTensor
    
    # Initialize distributed backend
    initialize()
    
    devices = [f"cuda:{i}" for i in range(num_gpus)]
    mesh = DeviceMesh(
        shape=(num_gpus,),
        axis_names=["model"],
        devices=devices
    )
    
    print(f"\nDeviceMesh created: shape={mesh.shape}")
    
    # Create LayoutMap for model parallelism
    layout_map = LayoutMap(mesh)
    
    # First, let's print all variable paths WITHOUT any sharding
    # to see the actual naming convention
    import keras_hub
    import keras
    
    print("\n" + "="*70)
    print("PHASE 1: PRINTING ALL VARIABLE PATHS (NO SHARDING)")
    print("="*70)
    
    with ModelParallel(
        layout_map=LayoutMap(mesh),  # Empty layout map - no sharding
        batch_dim_name="data",
        auto_shard_dataset=False
    ).scope():
        model = keras_hub.models.BertTextClassifier.from_preset(
            "bert_tiny_en_uncased",
            num_classes=2
        )
        
        print(f"\nVariable paths in keras_hub BERT model:")
        print("-" * 50)
        for v in model.variables:
            print(f"  {v.path}")
        print("-" * 50)
    
    print("\n" + "="*70)
    print("PHASE 2: TESTING SHARDING WITH CORRECTED PATTERNS")
    print("="*70)
    
    # Now create layout map with CORRECTED patterns that match keras_hub paths
    # The paths use underscore: token_embedding_embeddings, transformer_layer_0_attention_query_kernel
    layout_map_corrected = LayoutMap(mesh)
    
    # Use patterns that match keras_hub's underscore naming convention
    layout_map_corrected[".*token_embedding.*embeddings"] = ("model",)
    layout_map_corrected[".*position_embedding.*embeddings"] = ("model",)
    layout_map_corrected[".*segment_embedding.*embeddings"] = ("model",)
    layout_map_corrected[".*attention.*query.*kernel"] = ("model",)
    layout_map_corrected[".*attention.*key.*kernel"] = ("model",)
    layout_map_corrected[".*attention.*value.*kernel"] = ("model",)
    layout_map_corrected[".*attention.*output.*kernel"] = ("model",)
    layout_map_corrected[".*feedforward.*intermediate.*kernel"] = ("model",)
    layout_map_corrected[".*feedforward.*output.*kernel"] = ("model",)
    layout_map_corrected[".*pooled_dense.*kernel"] = ("model",)
    layout_map_corrected[".*logits.*kernel"] = ("model",)
    
    print("\nLayoutMap patterns (corrected for keras_hub):")
    for k in layout_map_corrected._layout_map:
        print(f"  Pattern '{k}' -> axes={layout_map_corrected._layout_map[k].axes}")
    
    # Build model WITH sharding
    with ModelParallel(
        layout_map=layout_map_corrected,
        batch_dim_name="data",
        auto_shard_dataset=False
    ).scope():
        model = keras_hub.models.BertTextClassifier.from_preset(
            "bert_tiny_en_uncased",
            num_classes=2
        )
        
        print(f"\n✓ Model loaded with sharding")
        
        # Count parameters
        total_params = sum(
            np.prod(w.shape) 
            for layer in model.layers 
            for w in (layer.weights if hasattr(layer, 'weights') else [])
        )
        print(f"Total parameters: {total_params:,}")
        
        # Verify sharding
        print(f"\nVerifying DTensor sharding:")
        print("-" * 50)
        
        sharded_count = 0
        replicated_count = 0
        
        for v in model.variables:
            # Check if the variable is a DTensor
            weight_tensor = None
            
            # Check direct DTensor
            if isinstance(v, DTensor):
                weight_tensor = v
            # Check wrapped DTensor
            elif hasattr(v, 'data') and isinstance(v.data, DTensor):
                weight_tensor = v.data
            # Check _torch attribute
            elif hasattr(v, '_torch') and isinstance(getattr(v, '_torch', None), DTensor):
                weight_tensor = getattr(v, '_torch')
            
            if weight_tensor is not None:
                local_shape = tuple(weight_tensor.to_local().shape)
                global_shape = tuple(v.shape)
                placements = weight_tensor.placements
                
                if local_shape != global_shape:
                    print(f"✓ SHARDED: {v.path}")
                    print(f"    Global: {global_shape} -> Local: {local_shape}")
                    print(f"    Placements: {placements}")
                    sharded_count += 1
                else:
                    print(f"  Replicated: {v.path}")
                    print(f"    Shape: {global_shape}")
                    replicated_count += 1
            else:
                print(f"  Regular tensor: {v.path}")
                print(f"    Shape: {tuple(v.shape)}")
        
        print("-" * 50)
        print(f"\nSharding Summary:")
        print(f"  Sharded weights: {sharded_count}")
        print(f"  Replicated weights: {replicated_count}")
        
        # Forward pass test
        print(f"\n" + "="*70)
        print("TEST: FORWARD PASS")
        print("="*70)
        
        batch_size = 2
        texts = ["This is a sample text for testing"] * batch_size
        
        try:
            # Prepare inputs
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
            
            print(f"Input shapes: token_ids={token_ids['token_ids'].shape}")
            
            outputs = model(token_ids, training=False)
            
            print(f"✓ Forward pass successful!")
            print(f"Output shape: {outputs.shape}")
            
        except Exception as e:
            print(f"✗ Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"\n" + "="*70)
        print("RESULT")
        print("="*70)
        if sharded_count > 0:
            print("✓ SHARDING VERIFIED!")
        else:
            print("✗ NO SHARDING APPLIED - Patterns need adjustment")


if __name__ == "__main__":
    run_hybrid_dp_mp_test()

