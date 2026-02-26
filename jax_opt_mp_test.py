import os

os.environ["KERAS_BACKEND"] = "jax"
# Use this to avoid JAX pre-allocating all GPU memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import keras
keras.config.disable_traceback_filtering()
from keras.src import ops
from keras.src import distribution
import numpy as np

def setup_dist():
    if jax.process_count() > 1:
        print(f"JAX Distributed initialized: {jax.process_count()} processes, {jax.device_count()} total devices")
    
    print(f"Process Index: {jax.process_index()}, Local Device Count: {jax.local_device_count()}, Total Device Count: {jax.device_count()}")
    print(f"Available Devices: {jax.devices()}")

def test_opt_model_parallel():
    setup_dist()
    print("\ntest started: test_opt_model_parallel")
    
    # Define mesh and layout map
    # Use jax.device_count() to allow sharding across all visible devices in a single process
    world_size = jax.device_count()
    mesh = distribution.DeviceMesh(shape=(world_size,), axis_names=("model",))
    print(f"Created device mesh: {mesh}")
    print(f"Mesh devices: {mesh.devices}")
    
    # Simple layout map for OPT:
    layout_map = distribution.LayoutMap(mesh)
    
    # Use regex-style matching for robustness
    layout_map[".*token_embedding/embeddings"] = (None, "model")
    layout_map[".*position_embedding/embeddings"] = (None, "model")
    
    # For Attention: Sharding the hidden_dim (dim 0) of QKV kernels
    layout_map[".*query/kernel"] = ("model", None, None)
    layout_map[".*key/kernel"] = ("model", None, None)
    layout_map[".*value/kernel"] = ("model", None, None)
    
    # Row parallel for out_proj (shard input dimension)
    layout_map[".*attention_output/kernel"] = ("model", None, None)
    
    # MLP sharding:
    layout_map[".*ffn_inner/kernel"] = (None, "model")
    layout_map[".*ffn_outer/kernel"] = ("model", None)
    
    print("\nConfigured LayoutMap:")
    for key, value in layout_map._layout_map.items():
        print(f"  {key}: {value}")

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
    rank = jax.process_index()
    print(f"\n[Process {rank}] Verifying weight sharding (Global vs Local):")
    for v in backbone.weights:
        if any(name in v.path for name in ["query/kernel", "token_embedding/embeddings", "ffn_inner/kernel", "ffn_outer/kernel"]):
            val = v.value
            if hasattr(val, "sharding"):
                print(f"[Process {rank}] Variable {v.path}:")
                print(f"  - Global shape: {tuple(val.shape)}")
                # Loop through all local devices this process manages
                for i in range(jax.local_device_count()):
                    try:
                        local_chunk = val.addressable_data(i)
                        device = jax.local_devices()[i]
                        print(f"  - Local shape on {device}: {tuple(local_chunk.shape)}")
                    except Exception:
                        pass
                print(f"  - Sharding:     {val.sharding}")
            else:
                print(f"[Process {rank}] Variable {v.path}: is_jax_array=False, shape={tuple(val.shape)}")

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
    # Provide data subset based on rank if multi-process
    process_count = jax.process_count()
    
    # Use a larger batch size total and slice it
    total_batch = batch_size * process_count
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
        causal_lm.fit(x, y, epochs=10, batch_size=batch_size)
        if rank == 0:
            print("model.fit completed successfully!")
    except Exception as e:
        print(f"model.fit failed: {e}")
        import traceback
        traceback.print_exc()

    # Test generation
    print("\nTesting model.generate...")
    try:
        # Generate some text
        prompt = {
            "token_ids": np.random.randint(0, 50272, (batch_size, 8)).astype("int32"),
            "padding_mask": np.ones((batch_size, 8), dtype="int32"),
        }
        generated = causal_lm.generate(prompt, max_length=12, stop_token_ids=None)
        if rank == 0:
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
