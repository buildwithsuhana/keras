import os

# Force Keras to use Torch backend
os.environ["KERAS_BACKEND"] = "torch"

# Prevent TensorFlow from grabbing all GPU memory if it gets imported
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import torch
import torch.distributed as dist
import keras
import numpy as np

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

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
            torch.cuda.set_device(local_rank)
        
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
    
    model_parallel = distribution.ModelParallel(layout_map=layout_map)
    print("Created ModelParallel distribution")
    
    print("Creating OPT backbone under distribution scope...")
    with model_parallel.scope():
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
    
    inputs = {
        "token_ids": token_ids,
        "padding_mask": padding_mask,
    }
    
    print("\nRunning test call...")
    try:
        output = backbone(inputs)
        if rank == 0:
            print(f"Output shape: {output.shape}")
    except Exception as e:
        print(f"Backbone call failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test model.fit
    from keras_hub.models import OPTCausalLM
    with model_parallel.scope():
        causal_lm = OPTCausalLM(backbone=backbone)

    print("\nTesting model.fit...")
    total_batch = batch_size * world_size
    x_full = {
        "token_ids": np.random.randint(0, 50272, (total_batch, seq_len)).astype("int32"),
        "padding_mask": np.ones((total_batch, seq_len), dtype="int32"),
    }
    y_full = np.random.randint(0, 50272, (total_batch, seq_len)).astype("int32")
    
    x = {
        "token_ids": x_full["token_ids"][rank*batch_size:(rank+1)*batch_size],
        "padding_mask": x_full["padding_mask"][rank*batch_size:(rank+1)*batch_size],
    }
    y = y_full[rank*batch_size:(rank+1)*batch_size]
    
    try:
        causal_lm.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
        verbose = 1 if rank == 0 else 0
        causal_lm.fit(x, y, epochs=2, batch_size=batch_size, verbose=verbose)
        if rank == 0:
            print("model.fit completed successfully!")
    except Exception as e:
        print(f"model.fit failed: {e}")
        import traceback
        traceback.print_exc()

    dist.destroy_process_group()

if __name__ == "__main__":
    test_opt_model_parallel()
