import os
os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import keras
from shared_autosharding import get_model, get_dataset, save_results

# Reproducibility
keras.utils.set_random_seed(42)

def run():
    from keras.src.distribution.distribution_lib import AutoTPDistribution, DeviceMesh, list_devices
    from keras.src.distribution.tensor_parallel.tensor_parallel import TensorParallelKeras
    
    x_raw, y_raw = get_dataset()
    x = {k: np.array(v) for k, v in x_raw.items()}
    if "padding_mask" in x:
        x["padding_mask"] = x["padding_mask"].astype("bool")
    y = np.array(y_raw)
    
    devices = list_devices("cpu")
    # Use 1D mesh for TP only to avoid JAX/Keras mesh conflicts
    device_mesh = DeviceMesh(shape=(2,), axis_names=("model",), devices=devices[:2])
    
    # 1. Create sacrificial model to generate the sharding layout map
    base_model = get_model()
    _ = base_model(x)
    
    # 2. Create the distribution object which computes the LayoutMap
    distribution = AutoTPDistribution(base_model, device_mesh=device_mesh)
    
    # 3. SET GLOBAL DISTRIBUTION BEFORE creating the real training model
    keras.distribution.set_distribution(distribution)
    
    # 4. Create the actual training model
    train_model = get_model()
    _ = train_model(x)
    
    # 5. Load weights
    train_model.set_weights(base_model.get_weights())
    
    # 6. Wrap in TensorParallelKeras
    tp_model = TensorParallelKeras(
        train_model, 
        device_count=2, 
        device_ids=devices[:2]
    )
    
    tp_model.compile(
        optimizer=keras.optimizers.SGD(1e-5),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    )

    # Initial Loss
    logits = tp_model(x)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    initial_loss = float(keras.ops.convert_to_numpy(loss_fn(y, logits)))
    print(f"JAX INITIAL_LOSS: {initial_loss:.12f}")

    # Train
    fit_loss = tp_model.train_on_batch(x, y)
    fit_loss_val = float(keras.ops.convert_to_numpy(fit_loss))
    print(f"JAX FIT_LOSS: {fit_loss_val:.12f}")
    
    # Final Loss
    logits_after = tp_model(x)
    final_loss = float(keras.ops.convert_to_numpy(loss_fn(y, logits_after)))
    print(f"JAX FINAL_LOSS: {final_loss:.12f}")
    
    save_results("jax", initial_loss, fit_loss_val, final_loss)

if __name__ == "__main__":
    run()
