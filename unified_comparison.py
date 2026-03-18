import os
import sys
import numpy as np
import keras

# 1. Environment and Backend Setup
backend = sys.argv[1]
os.environ["KERAS_BACKEND"] = backend

if backend == "jax":
    import jax
    jax.config.update("jax_enable_x64", True)

# Set global precision to float64 for bit-accuracy
keras.config.set_floatx("float64")

import torch
import keras_hub
from keras.src.distribution.distribution_lib import (
    DeviceMesh,
    LayoutMap,
    ModelParallel,
)

def run_training():
    # 2. Initialize Distributed Environment
    # On Kaggle, both JAX and Torch will detect the 2 GPUs
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
        model = keras_hub.models.OPTBackbone.from_preset("opt_125m_en")
        
        # Aggressively disable SDPA and Dropout for reproducibility
        for layer in model._flatten_layers(recursive=True):
            if hasattr(layer, "use_scaled_dot_product_attention"):
                layer.use_scaled_dot_product_attention = False
            if hasattr(layer, "_use_scaled_dot_product_attention"):
                layer._use_scaled_dot_product_attention = False
            if hasattr(layer, "_use_sdpa"):
                layer._use_sdpa = False
            if hasattr(layer, "dropout"):
                try: layer.dropout = 0.0
                except: pass
            if hasattr(layer, "dropout_rate"):
                layer.dropout_rate = 0.0
            if hasattr(layer, "rate"):
                layer.rate = 0.0

        # Use Adam with float64
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="mse",
        )

        # 6. Load Fixed Synthetic Data
        x = {
            "token_ids": np.load("data_cmp/x_token_ids.npy"),
            "padding_mask": np.load("data_cmp/x_padding_mask.npy"),
        }
        y = np.load("data_cmp/y.npy")

        # 7. Step 1: Capture Weight Updates
        target_weight = None
        for w in model.trainable_weights:
            if "embeddings" in w.path or "kernel" in w.path:
                if hasattr(w.value, "shape") and tuple(w.value.shape) != tuple(w.shape):
                    target_weight = w
                    break
        
        if target_weight is None:
             target_weight = model.trainable_weights[0]

        if rank == 0:
            print(f"[{backend}] Target weight for comparison: {target_weight.path}")

        weight_before = keras.ops.convert_to_numpy(target_weight.value).copy()

        # Run 1 step
        model.fit(x, y, epochs=1, steps_per_epoch=1, verbose=0)
        
        weight_after = keras.ops.convert_to_numpy(target_weight.value)
        update = weight_after - weight_before
        
        if rank == 0:
            np.save(f"update_step1_{backend}.npy", update)
            print(f"[{backend}] Step 1 update captured.")

        # 8. Complete Training (Total 10 Epochs)
        history_rest = model.fit(
            x, y,
            initial_epoch=1,
            epochs=10,
            batch_size=4,
            verbose=1 if rank == 0 else 0,
        )

        final_loss = keras.ops.convert_to_numpy(history_rest.history["loss"][-1])
        
        if rank == 0:
            with open(f"loss_{backend}.txt", "w") as f:
                f.write(str(float(final_loss)))
            print(f"[{backend}] Final loss: {final_loss}")

if __name__ == "__main__":
    try:
        run_training()
    except Exception as e:
        print(f"[{backend}] Rank {os.environ.get('RANK', '0')} failed: {e}")
        import traceback
        traceback.print_exc()
        os._exit(1)
