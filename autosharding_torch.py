import os
import sys
import torch

def _worker(rank, world_size, port):
    # Setup environment BEFORE Keras/Torch imports
    os.environ["KERAS_BACKEND"] = "torch"
    os.environ["KERAS_TORCH_DEVICE"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    
    import torch
    if hasattr(torch.backends, "mps"):
        torch.backends.mps.is_available = lambda: False

    import keras
    from shared_autosharding import get_model, get_dataset, save_results
    from keras.src.distribution.distribution_lib import AutoTPDistribution, DeviceMesh, initialize

    # Set seed in every process
    keras.utils.set_random_seed(42)
    initialize()
    
    x_raw, y_raw = get_dataset()
    x = {k: np.array(v) for k, v in x_raw.items()}
    y = np.array(y_raw)

    devices = [f"cpu:{i}" for i in range(world_size)]
    # Consistent 1D mesh
    device_mesh = DeviceMesh(shape=(world_size,), axis_names=("model",), devices=devices)
    
    # 1. Sacrificial model
    base_model = get_model()
    _ = base_model(x)
    
    # 2. Distribution
    distribution = AutoTPDistribution(base_model, device_mesh=device_mesh)
    
    # 3. Global Distribution
    keras.distribution.set_distribution(distribution)
    
    # 4. Actual model
    train_model = get_model()
    _ = train_model(x)
    
    from keras.src.distribution.tensor_parallel.tensor_parallel import TensorParallelKeras
    tp_model = TensorParallelKeras(
        train_model, 
        device_count=world_size, 
        device_ids=devices
    )
    
    tp_model.compile(
        optimizer=keras.optimizers.SGD(1e-5),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    )

    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    # Inference with no_grad
    with torch.no_grad():
        logits = tp_model(x)
        initial_loss = float(keras.ops.convert_to_numpy(loss_fn(y, logits)))
    
    # Train
    fit_loss = tp_model.train_on_batch(x, y)
    fit_loss_val = float(keras.ops.convert_to_numpy(fit_loss))
    
    with torch.no_grad():
        logits_after = tp_model(x)
        final_loss = float(keras.ops.convert_to_numpy(loss_fn(y, logits_after)))

    if rank == 0:
        print(f"TORCH INITIAL_LOSS: {initial_loss:.12f}")
        print(f"TORCH FIT_LOSS: {fit_loss_val:.12f}")
        print(f"TORCH FINAL_LOSS: {final_loss:.12f}")
        save_results("torch", initial_loss, fit_loss_val, final_loss)

def run():
    import socket
    def find_free_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]
    
    port = find_free_port()
    torch.multiprocessing.spawn(_worker, args=(2, port), nprocs=2, join=True)

if __name__ == "__main__":
    run()
