"""Test hybrid Data Parallel + Model Parallel with FIXED input conversion.

This version properly converts inputs to DTensors when model has sharded weights.
"""

import os
# Must be set before any other imports
os.environ["KERAS_BACKEND"] = "torch"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import numpy as np


def run_hybrid_dp_mp_test():
    """Run the hybrid DP+MP test with proper input conversion."""
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    print(f"\n{'='*70}")
    print(f"TEST: HYBRID DATA PARALLEL + MODEL PARALLEL")
    print(f"{'='*70}")
    print(f"Local rank: {local_rank}, World size: {world_size}")
    
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        gpu_id = local_rank % num_gpus
        torch.cuda.set_device(gpu_id)
        print(f"Process {local_rank} -> GPU {gpu_id}")
    
    # Initialize distributed backend
    from keras.src.distribution import DeviceMesh, LayoutMap, ModelParallel, initialize
    from keras.src.backend.torch import distribution_lib
    initialize()
    
    # Create DeviceMesh
    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    mesh = DeviceMesh(
        shape=(len(devices),),
        axis_names=["model"],
        devices=devices
    )
    
    print(f"\n[Rank {local_rank}] DeviceMesh: shape={mesh.shape}")
    
    # Create LayoutMap with CORRECTED patterns for BERT
    layout_map = LayoutMap(mesh)
    
    # FFN layers - these can be safely sharded
    layout_map[".*feedforward.*intermediate.*kernel"] = ("model",)
    layout_map[".*feedforward.*output.*kernel"] = ("model",)
    
    # Attention - replicate (can't shard due to reshape requirements)
    layout_map[".*attention.*query.*kernel"] = ()  # Replicate
    layout_map[".*attention.*key.*kernel"] = ()   # Replicate
    layout_map[".*attention.*value.*kernel"] = ()  # Replicate
    layout_map[".*attention.*output.*kernel"] = () # Replicate
    
    # Embeddings - replicate
    layout_map["token_embedding/embeddings"] = ()
    layout_map["position_embedding/embeddings"] = ()
    layout_map["segment_embedding/embeddings"] = ()
    
    # Classifier layers - replicate
    layout_map[".*pooled_dense.*kernel"] = ()
    layout_map[".*logits.*kernel"] = ()
    
    print(f"[Rank {local_rank}] LayoutMap patterns configured (FFN sharding)")
    
    # Create strategy
    strategy = ModelParallel(
        layout_map=layout_map,
        batch_dim_name="data",
        auto_shard_dataset=False
    )
    
    # Build model
    import keras_hub
    import keras
    
    with strategy.scope():
        print(f"\n[Rank {local_rank}] Loading BERT-tiny model...")
        model = keras_hub.models.BertTextClassifier.from_preset(
            "bert_tiny_en_uncased",
            num_classes=2
        )
        print(f"[Rank {local_rank}] ✓ Model loaded")
        
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
    
    from torch.distributed._tensor import DTensor
    
    sharded_count = 0
    replicated_count = 0
    
    for v in model.variables:
        torch_tensor = None
        
        if hasattr(v, 'value') and hasattr(v.value, 'data'):
            torch_tensor = v.value.data
        elif hasattr(v, 'value'):
            torch_tensor = v.value
        else:
            torch_tensor = v
        
        is_dtensor = False
        
        if isinstance(torch_tensor, DTensor):
            is_dtensor = True
            local_shape = tuple(torch_tensor.to_local().shape)
            global_shape = tuple(torch_tensor.shape)
        elif hasattr(torch_tensor, '_torch'):
            wrapped = getattr(torch_tensor, '_torch', None)
            if isinstance(wrapped, DTensor):
                is_dtensor = True
                local_shape = tuple(wrapped.to_local().shape)
                global_shape = tuple(torch_tensor.shape)
        
        if is_dtensor:
            if local_shape != global_shape:
                print(f"[Rank {local_rank}] ✓ SHARDED: {v.path}")
                print(f"    Global: {global_shape} -> Local: {local_shape}")
                sharded_count += 1
            else:
                print(f"[Rank {local_rank}]   Replicated: {v.path}")
                replicated_count += 1
    
    print(f"\n[Rank {local_rank}] Sharding Summary:")
    print(f"  Sharded: {sharded_count}, Replicated: {replicated_count}")
    
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
        
        # CRITICAL: Convert inputs to DTensors when model has sharded weights!
        # This is required by PyTorch DTensor for mixed operations
        token_ids = distribution_lib.prepare_input_for_distribution(token_ids)
        
        print(f"[Rank {local_rank}] Inputs converted to DTensors")
        
        outputs = model(token_ids, training=False)
        
        print(f"[Rank {local_rank}] ✓ Forward pass successful!")
        print(f"Output shape: {outputs.shape}")
        
    except Exception as e:
        print(f"[Rank {local_rank}] ✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Cleanup
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    
    print(f"\n{'='*70}")
    print("TEST COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    run_hybrid_dp_mp_test()

