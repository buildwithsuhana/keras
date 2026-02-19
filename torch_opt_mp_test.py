import os

os.environ["KERAS_BACKEND"] = "torch"
os.environ["KERAS_TORCH_DEVICE"] = "cpu"

import torch
import torch.distributed as dist
import keras
keras.config.disable_traceback_filtering()
from keras.src import ops
from keras.src import distribution
import numpy as np

def setup_dist():
    if not dist.is_initialized():
        if "RANK" not in os.environ:
            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["MASTER_PORT"] = "12355"
            os.environ["RANK"] = "0"
            os.environ["WORLD_SIZE"] = "1"
            # On macOS, lo0 is the loopback interface
            os.environ["GLOO_SOCKET_IFNAME"] = "lo0"
        
        print(f"Initializing process group (RANK={os.environ.get('RANK')}, WORLD_SIZE={os.environ.get('WORLD_SIZE')})...")
        keras.distribution.initialize()
    print(f"World size: {dist.get_world_size()}, Rank: {dist.get_rank()}")

def test_opt_model_parallel():
    setup_dist()
    print("test started: test_opt_model_parallel")
    # Define mesh and layout map
    # 1D mesh for model parallel (sharding weights)
    world_size = dist.get_world_size()
    mesh = distribution.DeviceMesh(shape=(world_size,), axis_names=("model",))
    print(f"Created device mesh: {mesh}")
    # Simple layout map for OPT: shard embeddings and some dense layers
    layout_map = distribution.LayoutMap(mesh)
    layout_map["token_embedding/embeddings"] = (None, "model")
    # Shard the projection layers in attention
    # OPT projection
    #  weights are often (hidden_dim, hidden_dim)
    layout_map["query/kernel"] = (None, "model")
    layout_map["key/kernel"] = (None, "model")
    layout_map["value/kernel"] = (None, "model")
    layout_map["out_proj/kernel"] = ("model", None)
    # Shard the MLP layers
    layout_map["ffn_inner/kernel"] = (None, "model")
    layout_map["ffn_outer/kernel"] = ("model", None)
    print(f"Defined layout map: {layout_map}")
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
    print("\nVerifying weight sharding (first few weights):")
    for v in backbone.weights[:5]:
        is_dtensor = hasattr(v.value, "device_mesh")
        sharding = v.value.placements if is_dtensor else "N/A"
        print(f"Variable {v.path}: is_dtensor={is_dtensor}, shape={v.shape}, sharding={sharding}")

    # Test call
    print("\nRunning test call...")
    batch_size = 2
    seq_len = 32
    token_ids = np.random.randint(0, 50272, (batch_size, seq_len)).astype("int32")
    padding_mask = np.ones((batch_size, seq_len), dtype="int32")
    
    inputs = {
        "token_ids": token_ids,
        "padding_mask": padding_mask,
    }
    
    output = backbone(inputs)
    print(f"Output shape: {output.shape}")
    
    # Test model.fit
    from keras_hub.models import OPTCausalLM
    with model_parallel.scope():
        causal_lm = OPTCausalLM(backbone=backbone)

    print("\nTesting model.fit...")
    x = {
        "token_ids": np.random.randint(0, 50272, (batch_size, seq_len)).astype("int32"),
        "padding_mask": np.ones((batch_size, seq_len), dtype="int32"),
    }
    y = np.random.randint(0, 50272, (batch_size, seq_len)).astype("int32")
    
    causal_lm.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    causal_lm.fit(x, y, epochs=1, batch_size=batch_size)
    print("model.fit completed successfully!")

    # Test generation
    print("\nTesting model.generate...")
    try:
        # Generate some text
        prompt = {
            "token_ids": np.random.randint(0, 50272, (batch_size, 8)).astype("int32"),
            "padding_mask": np.ones((batch_size, 8), dtype="int32"),
        }
        generated = causal_lm.generate(prompt, max_length=12, stop_token_ids=None)
        if isinstance(generated, dict):
            print(f"Generated token_ids shape: {generated['token_ids'].shape}")
        else:
            print(f"Generated shape: {generated.shape}")
        print("model.generate completed successfully!")
    except Exception as e:
        print(f"model.generate failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_opt_model_parallel()
