import os

# Force Keras to use Torch backend
os.environ["KERAS_BACKEND"] = "torch"

# Isolate GPUs for each rank and hide them from TF to avoid hangs/conflicts
if "LOCAL_RANK" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["LOCAL_RANK"]
    # Force Keras to use the isolated GPU (it will always be index 0 due to isolation)
    os.environ["KERAS_TORCH_DEVICE"] = "cuda:0"

# Prevent TensorFlow from grabbing all GPU memory if it gets imported
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import torch
import torch.distributed as dist
import keras
import numpy as np

keras.config.disable_traceback_filtering()
from keras.src import distribution

def setup_dist():
    if not dist.is_initialized():
        if "RANK" not in os.environ:
            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["MASTER_PORT"] = "12355"
            os.environ["RANK"] = "0"
            os.environ["WORLD_SIZE"] = "1"
        
        # Determine backend based on availability
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        
        if backend == "nccl":
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            torch.cuda.set_device(0) # Always 0 due to isolation
        
        print(f"Initializing process group (RANK={os.environ.get('RANK')}, WORLD_SIZE={os.environ.get('WORLD_SIZE')}, BACKEND={backend})...")
        dist.init_process_group(backend=backend)
        
    print(f"World size: {dist.get_world_size()}, Rank: {dist.get_rank()}, Current Device: {torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'}")

def test_opt_model_parallel():
    setup_dist()
    print("test started: test_opt_model_parallel")
    
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    # Define mesh and layout map
    mesh = distribution.DeviceMesh(shape=(world_size,), axis_names=("model",))
    print(f"Created device mesh: {mesh}")
    
    layout_map = distribution.LayoutMap(mesh)
    
    # Sharding layout for OPT
    layout_map[".*token_embedding/embeddings"] = (None, "model")
    layout_map[".*position_embedding/embeddings"] = (None, "model")
    layout_map[".*query/kernel"] = ("model", None, None)
    layout_map[".*key/kernel"] = ("model", None, None)
    layout_map[".*value/kernel"] = ("model", None, None)
    layout_map[".*attention_output/kernel"] = ("model", None, None)
    layout_map[".*ffn_inner/kernel"] = (None, "model")
    layout_map[".*ffn_outer/kernel"] = ("model", None)
    
    model_parallel = distribution.ModelParallel(layout_map=layout_map, auto_shard_dataset=False)
    print("Created ModelParallel distribution")
    distribution.set_distribution(model_parallel)
    
    print("Creating OPT backbone under distribution scope...")
    from keras_hub.models import OPTBackbone
    # Smallest OPT for testing
    backbone = OPTBackbone(
        vocabulary_size=50272,
        num_layers=1,
        num_heads=12,
        hidden_dim=768,
        intermediate_dim=3072,
        max_sequence_length=2048,
    )

    # Verify sharding
    print(f"\n[Rank {rank}] Verifying weight sharding:")
    for v in backbone.weights:
        if any(name in v.path for name in ["query/kernel", "token_embedding/embeddings"]):
            val = v.value
            from torch.distributed.tensor import DTensor
            if isinstance(val, DTensor):
                local_shape = val.to_local().shape
                print(f"[Rank {rank}] Variable {v.path}:")
                print(f"  - Global shape: {tuple(val.shape)}")
                print(f"  - Local shape:  {tuple(local_shape)}")
                print(f"  - Placements:   {val.placements}")
            else:
                print(f"[Rank {rank}] Variable {v.path}: [NOT SHARDED] shape={tuple(val.shape)}")

    # Test call
    batch_size = 2
    seq_len = 32
    token_ids = np.random.randint(0, 50272, (batch_size, seq_len)).astype("int32")
    padding_mask = np.ones((batch_size, seq_len), dtype="int32")
    
    # Pre-distribute inputs to test DTensor input handling
    from keras.src.backend.torch.distribution_lib import distribute_tensor
    # Shard batch on 'model' axis
    data_layout = distribution.TensorLayout(("model", None), mesh)
    
    token_ids_dt = distribute_tensor(torch.as_tensor(token_ids), data_layout)
    padding_mask_dt = distribute_tensor(torch.as_tensor(padding_mask), data_layout)
    
    inputs = {
        "token_ids": token_ids_dt,
        "padding_mask": padding_mask_dt,
    }
    
    print("\nRunning test call with DTensor inputs...")
    try:
        output = backbone(inputs)
        if rank == 0:
            print(f"Output shape: {output.shape}")
            print(f"Output type: {type(output)}")
    except Exception as e:
        print(f"Backbone call failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test model.fit
    from keras_hub.models import OPTCausalLM
    causal_lm = OPTCausalLM(backbone=backbone)

    print("\nTesting model.fit with DTensor inputs...")
    total_batch = batch_size * world_size
    
    # For model.fit, we can also pass DTensors directly
    x_dt = {
        "token_ids": distribute_tensor(
            torch.as_tensor(np.random.randint(0, 50272, (total_batch, seq_len)).astype("int32")),
            data_layout
        ),
        "padding_mask": distribute_tensor(
            torch.as_tensor(np.ones((total_batch, seq_len), dtype="int32")),
            data_layout
        ),
    }
    y_dt = distribute_tensor(
        torch.as_tensor(np.random.randint(0, 50272, (total_batch, seq_len)).astype("int32")),
        data_layout
    )
    
    try:
        causal_lm.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
        verbose = 1 if rank == 0 else 0
        causal_lm.fit(x_dt, y_dt, epochs=2, batch_size=total_batch, verbose=verbose)
        if rank == 0:
            print("model.fit completed successfully!")
    except Exception as e:
        print(f"model.fit failed: {e}")
        import traceback
        traceback.print_exc()

    distribution.set_distribution(None)
    dist.destroy_process_group()

if __name__ == "__main__":
    test_opt_model_parallel()
