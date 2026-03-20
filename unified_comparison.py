import os
import sys

# 1. Environment and Backend Setup
backend = sys.argv[1]
os.environ["KERAS_BACKEND"] = backend

import numpy as np
import keras

# Set global precision to float32 (default)
keras.config.set_floatx("float32")

import torch
import keras_hub
from keras.src.distribution.distribution_lib import (
    DeviceMesh,
    LayoutMap,
    ModelParallel,
)

def run_training():
    # 2. Initialize Distributed Environment
    if backend == "torch":
        keras.distribution.initialize()
    
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "2"))
    
    if rank == 0:
        print(f"[{backend}] Starting ModelParallel parity test on {world_size} GPUs...")

    # 3. Deterministic Seeds
    keras.utils.set_random_seed(42)

    # 4. Define Distribution Strategy
    devices = keras.distribution.list_devices("gpu")
    if len(devices) < 2:
        # Fallback for testing if GPUs are not present
        devices = keras.distribution.list_devices("cpu")
        
    mesh = DeviceMesh(
        shape=(2,), 
        axis_names=("model",), 
        devices=devices[:2]
    )
    
    layout_map = LayoutMap(mesh)
    # Shard ALL major layers
    layout_map[".*embeddings.*"] = ("model", None)
    layout_map[".*query_dense.*kernel"] = (None, "model")
    layout_map[".*key_dense.*kernel"] = (None, "model")
    layout_map[".*value_dense.*kernel"] = (None, "model")
    layout_map[".*attention.*output_dense.*kernel"] = ("model", None)
    layout_map[".*ffn_intermediate.*kernel"] = (None, "model")
    layout_map[".*ffn_output.*kernel"] = ("model", None)
    
    distribution = ModelParallel(layout_map=layout_map, batch_dim_name="model")

    # 5. Build and Compile Model
    with distribution.scope():
        # Force float32 preset loading
        model = keras_hub.models.OPTBackbone.from_preset("opt_125m_en", dtype="float32")
        
        # Aggressively disable SDPA and Dropout for reproducibility
        for layer in model._flatten_layers(recursive=True):
            if hasattr(layer, "use_scaled_dot_product_attention"):
                layer.use_scaled_dot_product_attention = False
            if hasattr(layer, "dropout"):
                try: layer.dropout = 0.0
                except: pass

        # Synchronize initial weights
        weights_file = "initial_weights_mp.weights.h5"
        if backend == "jax":
            model.save_weights(weights_file)
            if rank == 0:
                print(f"[{backend}] Initial weights saved.")
        else:
            model.load_weights(weights_file)
            if rank == 0:
                print(f"[{backend}] Initial weights loaded.")
            
            # --- PYTORCH SHARDING VERIFICATION ---
            from torch.distributed.tensor import DTensor
            sharded_weights = []
            for w in model.trainable_weights:
                if isinstance(w.value, DTensor):
                    # Check if any placement is Shard
                    is_sharded = any(p.is_shard() for p in w.value.placements)
                    if is_sharded:
                        sharded_weights.append(w.path)
            
            if rank == 0:
                print(f"[{backend}] SHARDING VERIFICATION:")
                print(f"[{backend}] Total weights: {len(model.trainable_weights)}")
                print(f"[{backend}] Sharded weights (DTensor): {len(sharded_weights)}")
                if len(sharded_weights) > 0:
                    sample_w = model.trainable_weights[1].value # Pick a kernel
                    print(f"[{backend}] Sample weight ({model.trainable_weights[1].path}):")
                    print(f"[{backend}]   - Global shape: {sample_w.shape}")
                    print(f"[{backend}]   - Placements: {sample_w.placements}")
                    print(f"[{backend}]   - Local shape: {sample_w.to_local().shape}")
            # ---------------------------------------

        # Use Adam with explicit epsilon and NO JIT
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001, epsilon=1e-7),
            loss="mse",
            jit_compile=False,
        )

        # 6. Load Fixed Synthetic Data
        x = {
            "token_ids": np.load("data_cmp/x_token_ids.npy"),
            "padding_mask": np.load("data_cmp/x_padding_mask.npy"),
        }
        y = np.load("data_cmp/y.npy")

        # Step 0: Compare initial loss to verify sync
        # We must calculate GLOBAL loss to match JAX
        out0 = model.predict(x, batch_size=8, verbose=0)
        
        y_local = y
        if backend == "torch":
            start = rank * 4
            end = (rank + 1) * 4
            y_local = y[start:end]
            
        local_loss = np.mean(np.square(out0 - y_local))
        
        if backend == "torch":
            # Sum local losses and divide by world_size to get global mean
            t_loss = torch.tensor(local_loss, device=torch_core.get_device())
            torch.distributed.all_reduce(t_loss, op=torch.distributed.ReduceOp.SUM)
            initial_loss = t_loss.item() / world_size
        else:
            initial_loss = local_loss

        if rank == 0:
            print(f"[{backend}] Initial Loss (Step 0, Global): {initial_loss:.12f}")
            with open(f"initial_loss_{backend}.txt", "w") as f:
                f.write(str(float(initial_loss)))

        # 7. Step 1: Capture Weight Updates
        target_weight = None
        for w in model.trainable_weights:
            if "kernel" in w.path:
                # Pick a weight that is actually sharded
                if hasattr(w.value, "shape") and tuple(w.value.shape) != tuple(w.shape):
                    target_weight = w
                    break
        
        if target_weight is None:
             target_weight = model.trainable_weights[0]

        if rank == 0:
            print(f"[{backend}] Target weight for comparison: {target_weight.path}")

        weight_before = keras.ops.convert_to_numpy(target_weight.value).copy()

        # Run 1 step with shuffle=False
        model.fit(x, y, epochs=1, steps_per_epoch=1, batch_size=8, shuffle=False, verbose=0)
        
        weight_after = keras.ops.convert_to_numpy(target_weight.value)
        update = weight_after - weight_before
        
        if rank == 0:
            np.save(f"update_step1_{backend}.npy", update)
            print(f"[{backend}] Step 1 update captured.")

        # 8. Complete Training (Total 10 Epochs) with shuffle=False
        history_rest = model.fit(
            x, y,
            initial_epoch=1,
            epochs=10,
            batch_size=8,
            shuffle=False,
            verbose=1 if rank == 0 else 0,
        )

        final_loss_local = history_rest.history["loss"][-1]
        if backend == "torch":
            t_loss = torch.tensor(final_loss_local, device=torch_core.get_device())
            torch.distributed.all_reduce(t_loss, op=torch.distributed.ReduceOp.SUM)
            final_loss = t_loss.item() / world_size
        else:
            final_loss = final_loss_local
        
        if rank == 0:
            with open(f"loss_{backend}.txt", "w") as f:
                f.write(str(float(final_loss)))
            print(f"[{backend}] Final loss (Global): {final_loss}")

if __name__ == "__main__":
    try:
        run_training()
    except Exception as e:
        print(f"[{backend}] Rank {os.environ.get('RANK', '0')} failed: {e}")
        import traceback
        traceback.print_exc()
        os._exit(1)
