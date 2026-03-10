import os
import torch  # Ensure torch is imported before usage

# Check if GPU is available and set the device accordingly
default_device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["KERAS_TORCH_DEVICE"] = default_device
os.environ["KERAS_BACKEND"] = "torch"
os.environ["KERAS_DISTRIBUTION_DEBUG"] = "1"

# Set NCCL environment variables to avoid conflicts
os.environ["NCCL_DEBUG"] = "WARN"
os.environ["NCCL_P2P_LEVEL"] = "NVL"
os.environ["NCCL_BLOCKING_WAIT"] = "1"

import numpy as np
import keras
import keras_hub
from keras.distribution import DeviceMesh, LayoutMap, ModelParallel, TensorLayout


def train_opt_model_parallel():

    # Initialize torch distributed process group
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")

    keras.distribution.initialize()

    # Setup DeviceMesh
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    devices = [f"cuda:{i}" for i in range(world_size)] if torch.cuda.is_available() else [f"cpu:{i}" for i in range(world_size)]
    mesh = DeviceMesh(shape=(world_size,), axis_names=("model",), devices=devices)

    # Ensure each process uses its assigned device
    torch.cuda.set_device(rank) if torch.cuda.is_available() else None

    tmp_model = keras_hub.models.OPTBackbone(
        vocabulary_size=1000, num_layers=2, num_heads=2, hidden_dim=64, intermediate_dim=128, max_sequence_length=32, dropout=0.0,
    )
    tmp_model.build({"token_ids": (None, 32), "padding_mask": (None, 32)})

    layout_map = LayoutMap(mesh)
    for var in tmp_model.variables:
        path = var.path
        axes = [None] * len(var.shape)
        
        # 1. Embeddings: Shard embedding dim
        if "token_embedding/embeddings" in path:
            axes[-1] = "model"
            
        # 2. FFN: Standard Colwise/Rowwise
        elif "feedforward_intermediate_dense/kernel" in path:
            axes[-1] = "model"
        elif "feedforward_intermediate_dense/bias" in path:
            axes[0] = "model"
        elif "feedforward_output_dense/kernel" in path:
            axes[0] = "model"
            
        # 3. LayerNorm: Shard hidden dim (Fixed in ops/nn.py)
        elif "layer_norm" in path and ("gamma" in path or "beta" in path):
            axes[0] = "model"
            
        # 4. Attention: Shard heads (Fixed in ops/nn.py)
        # Note: We can shard heads now because ops/nn.py uses decomposed SDPA
        elif "self_attention" in path and "/kernel" in path and len(var.shape) == 3:
            axes[1] = "model"
        elif "self_attention/attention_output/kernel" in path:
            axes[0] = "model"
        
        layout = TensorLayout(axes=tuple(axes), device_mesh=mesh)
        layout_map[path] = layout

    print(f"Constructed LayoutMap with {len(layout_map._layout_map)} entries")
    print(f"  Testing end-to-end with permanent backend fixes (ALL LAYERS SHARDED)")

    distribution = ModelParallel(layout_map=layout_map, auto_shard_dataset=False)

    # Build model INSIDE distribution scope
    print(f"Creating and building model within distribution scope on RANK {os.environ.get('RANK', '0')}...")
    with distribution.scope():
        model = keras_hub.models.OPTBackbone(
            vocabulary_size=1000, num_layers=2, num_heads=2, hidden_dim=64, intermediate_dim=128, max_sequence_length=32, dropout=0.0,
        )
        model.build({"token_ids": (None, 32), "padding_mask": (None, 32)})
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss=keras.losses.MeanSquaredError())
        
        # Prepare data and fit
        token_ids = np.random.randint(0, 1000, (16, 32)).astype("int32")
        padding_mask = np.ones((16, 32), dtype="int32")
        y = np.random.randn(16, 32, 64).astype("float32")
        x = {"token_ids": token_ids, "padding_mask": padding_mask}
        
        print(f"\nRunning model.fit() on RANK {os.environ.get('RANK', '0')}...")
        try:
            history = model.fit(x, y, epochs=1, batch_size=4, verbose=1)
            print(f"\n✓ model.fit() completed successfully on RANK {os.environ.get('RANK', '0')}!")
            final_loss = float(history.history['loss'][-1])
            print(f"  Final loss on RANK {os.environ.get('RANK', '0')}: {final_loss:.4f}")
        except Exception as e:
            print(f"\n✗ model.fit() failed on RANK {os.environ.get('RANK', '0')}: {e}")
            import traceback
            traceback.print_exc()
            
        # Validation
        print(f"Validating sharding on RANK {os.environ.get('RANK', '0')}...")
        shard_count = 0
        replicate_count = 0
        import torch.distributed.tensor as dt
        for v in model.variables:
            if hasattr(v.value, 'placements'):
                has_shard = any(isinstance(p, dt.Shard) for p in v.value.placements)
                if has_shard:
                    shard_count += 1
                    if os.environ.get('RANK', '0') == '0' and ("/kernel" in v.path or "layer_norm" in v.path or "token_embedding" in v.path):
                        print(f"  [SHARDED] {v.path}: {v.value.placements}")
                else: replicate_count += 1
            else: replicate_count += 1
        
        print(f"\nSharding Summary on RANK {os.environ.get('RANK', '0')}:")
        print(f"  Sharded variables: {shard_count}")
        print(f"  Replicated variables: {replicate_count}")

if __name__ == "__main__":
    train_opt_model_parallel()
