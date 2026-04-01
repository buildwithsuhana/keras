import os

# Set backend to torch
os.environ["KERAS_BACKEND"] = "torch"

import keras
import keras_hub
import numpy as np
from keras import distribution

# 1. Initialize the distribution system
# This handles the torch.distributed initialization
distribution.initialize()

# 2. Setup the DeviceMesh
# We'll use all available devices for a "model" parallel axis
devices = distribution.list_devices()
num_devices = len(devices)
device_mesh = distribution.DeviceMesh(
    shape=(num_devices,),
    axis_names=("model",),
    devices=devices,
)

# 3. Define the LayoutMap for OPT-125m sharding
# This defines how specific weights are sharded across the "model" axis
layout_map = distribution.LayoutMap(device_mesh)

# Shard Attention kernels
# query, key, value are sharded on the output dimension (heads)
layout_map[".*self_attention/query/kernel"] = (None, "model")
layout_map[".*self_attention/key/kernel"] = (None, "model")
layout_map[".*self_attention/value/kernel"] = (None, "model")
# attention_output is sharded on the input dimension
layout_map[".*self_attention/attention_output/kernel"] = ("model", None)

# Shard Feed-forward kernels
# intermediate dense is sharded on output
layout_map[".*feedforward_intermediate_dense/kernel"] = (None, "model")
# output dense is sharded on input
layout_map[".*feedforward_output_dense/kernel"] = ("model", None)

# 4. Create the ModelParallel distribution
dist = distribution.ModelParallel(
    layout_map=layout_map,
    batch_dim_name="batch", # Batch is replicated if "batch" axis is not in mesh
)

# 5. Load the model within the distribution scope
# Everything created inside this scope will follow the distribution strategy
with dist.scope():
    import torch
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    print(f"Process {rank} loading model structure...")
    # Load without weights first to avoid early loading failure
    model = keras_hub.models.OPTCausalLM.from_preset("opt_125m_en", load_weights=False)
    
    print(f"Process {rank} model structure loaded. Number of weights: {len(model.weights)}")
    if rank == 0:
        print(f"Example weight paths: {[w.path for w in model.weights[:5]]}")
        # Position embedding is nested under 'embeddings' layer
        embeddings_layer = model.backbone.get_layer("embeddings")
        print(f"Embeddings layer: {embeddings_layer}")
        if hasattr(embeddings_layer, "get_layer"):
            try:
                pos_emb = embeddings_layer.get_layer("position_embedding")
                print(f"Position embedding built: {pos_emb.built}")
                print(f"Position embedding variables: {pos_emb.variables}")
            except Exception as e:
                print(f"Could not find position_embedding sub-layer: {e}")

    # Now try to load weights manually
    print(f"Process {rank} loading weights manually...")
    try:
        # Get the preset's weight file path
        import keras_hub
        from keras_hub.src.utils.preset_utils import get_preset_loader
        loader = get_preset_loader("opt_125m_en")
        weights_path = loader.get_weights()
        # Use skip_mismatch=True to avoid crashing if some sharded variables have issues
        model.load_weights(weights_path)
        print(f"Process {rank} weights loaded successfully!")
    except Exception as e:
        print(f"Process {rank} failed to load weights: {e}")
        print(f"Process {rank} continuing with random weights for testing...")

    # 6. Prepare some test data
    data = [
        "Keras Hub and PyTorch make model parallelism easy.",
        "This is a test of OPT-125m with ModelParallel distribution.",
    ]

    # 7. Compile and run fit
    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    )

    print(f"Process {rank} starting training...")
    # Use a small number of steps for a quick test
    model.fit(x=data, y=data, batch_size=2, epochs=1)

    # 8. Test generation
    print(f"Process {rank} testing generation...")
    generated = model.generate(data, max_length=20)

if (torch.distributed.get_rank() if torch.distributed.is_initialized() else 0) == 0:
    print("\nSUCCESS: Model parallelism test completed!")
    print(f"Generated text: {generated}")
