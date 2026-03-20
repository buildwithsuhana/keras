import os
import sys

# 1. Environment and Backend Setup
backend = sys.argv[1]
os.environ["KERAS_BACKEND"] = backend

import numpy as np
import keras
import keras_hub
from keras.src.distribution.distribution_lib import DataParallel

def run_training():
    # 2. Initialize Distributed Environment
    # JAX doesn't need this for single-process, Torch does.
    if backend == "torch":
        keras.distribution.initialize()
    
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "2"))
    
    if rank == 0:
        print(f"[{backend}] Starting DataParallel parity test on {world_size} GPUs...")

    # 3. Deterministic Seeds
    keras.utils.set_random_seed(42)

    # 4. Define Distribution Strategy (DataParallel)
    devices = keras.distribution.list_devices("gpu")
    if len(devices) < 2:
        devices = keras.distribution.list_devices("cpu")
    distribution = DataParallel(devices=devices[:2])

    # 5. Build and Compile Model
    with distribution.scope():
        model = keras_hub.models.OPTBackbone.from_preset("opt_125m_en")
        
        # Disable SDPA and Dropout for reproducibility
        for layer in model._flatten_layers(recursive=True):
            if hasattr(layer, "use_scaled_dot_product_attention"):
                layer.use_scaled_dot_product_attention = False
            if hasattr(layer, "dropout"):
                try: layer.dropout = 0.0
                except: pass

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="mse",
        )

        # 6. Load Fixed Synthetic Data (Batch size 8)
        x_full = {
            "token_ids": np.load("data_cmp/x_token_ids.npy"),
            "padding_mask": np.load("data_cmp/x_padding_mask.npy"),
        }
        y_full = np.load("data_cmp/y.npy")

        # In DP, each process sees a slice of the global batch
        # World size is 2. Global batch is 8. Local is 4.
        start = rank * 4
        end = (rank + 1) * 4
        x = {k: v[start:end] for k, v in x_full.items()}
        y = y_full[start:end]

        if rank == 0:
            print(f"[{backend}] Data loaded. Local batch shape: {y.shape}")

        # 7. Capture Weight Updates
        # For DP, we'll pick the first kernel
        target_weight = None
        for w in model.trainable_weights:
            if "kernel" in w.path:
                target_weight = w
                break
        
        if rank == 0:
            print(f"[{backend}] Target weight for comparison: {target_weight.path}")

        weight_before = keras.ops.convert_to_numpy(target_weight.value).copy()

        # Run 1 step
        # Since we pass local data, we set steps_per_epoch=1
        model.fit(x, y, epochs=1, steps_per_epoch=1, verbose=0)
        
        weight_after = keras.ops.convert_to_numpy(target_weight.value)
        update = weight_after - weight_before
        
        if rank == 0:
            np.save(f"dp_update_step1_{backend}.npy", update)
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
            with open(f"dp_loss_{backend}.txt", "w") as f:
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
