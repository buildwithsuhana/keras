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
        # Force float32 preset loading
        model = keras_hub.models.OPTBackbone.from_preset("opt_125m_en", dtype="float32")
        
        # Disable SDPA and Dropout for reproducibility
        for layer in model._flatten_layers(recursive=True):
            if hasattr(layer, "use_scaled_dot_product_attention"):
                layer.use_scaled_dot_product_attention = False
            if hasattr(layer, "dropout"):
                try: layer.dropout = 0.0
                except: pass

        # Synchronize initial weights
        weights_file = "initial_weights_dp.weights.h5"
        if backend == "jax":
            model.save_weights(weights_file)
            if rank == 0:
                print(f"[{backend}] Initial weights saved.")
        else:
            model.load_weights(weights_file)
            if rank == 0:
                print(f"[{backend}] Initial weights loaded.")

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

        # Step 0: Compare initial loss
        out0 = model.predict(x, batch_size=8, verbose=0)
        
        # In multi-process (Torch), we only get local predictions
        y_local = y
        if backend == "torch":
            start = rank * 4
            end = (rank + 1) * 4
            y_local = y[start:end]
            
        initial_loss = np.mean(np.square(out0 - y_local))
        if rank == 0:
            print(f"[{backend}] Initial Loss (Step 0): {initial_loss:.12f}")
            with open(f"dp_initial_loss_{backend}.txt", "w") as f:
                f.write(str(float(initial_loss)))

        # 7. Capture Weight Updates
        target_weight = None
        for w in model.trainable_weights:
            if "kernel" in w.path:
                target_weight = w
                break
        
        if rank == 0:
            print(f"[{backend}] Target weight for comparison: {target_weight.path}")

        weight_before = keras.ops.convert_to_numpy(target_weight.value).copy()

        # Run 1 step with shuffle=False
        model.fit(x, y, epochs=1, steps_per_epoch=1, batch_size=8, shuffle=False, verbose=0)
        
        weight_after = keras.ops.convert_to_numpy(target_weight.value)
        update = weight_after - weight_before
        
        if rank == 0:
            np.save(f"dp_update_step1_{backend}.npy", update)
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
