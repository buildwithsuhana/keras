import os
import sys
import numpy as np

backend = os.environ.get("KERAS_BACKEND", "torch")

import keras
# Set seed for Keras
keras.utils.set_random_seed(42)
keras.config.disable_traceback_filtering()

from keras.src import ops
from keras.src import distribution
from keras_hub.models import OPTBackbone, OPTCausalLM

# Backend-specific imports for distribution info
if backend == "torch":
    import torch
    import torch.distributed as dist
    def get_rank():
        return dist.get_rank() if dist.is_initialized() else 0
    def setup_dist():
        if not dist.is_initialized():
            if "RANK" not in os.environ:
                os.environ["MASTER_ADDR"] = "127.0.0.1"
                os.environ["MASTER_PORT"] = "12355"
                os.environ["RANK"] = "0"
                os.environ["WORLD_SIZE"] = "1"
            dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
            if torch.cuda.is_available():
                torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
else:
    import jax
    def get_rank():
        return jax.process_index()
    def setup_dist():
        pass

def run_test():
    setup_dist()
    rank = get_rank()
    
    # 1. Setup Model Parallel
    if backend == "torch":
        world_size = dist.get_world_size() if dist.is_initialized() else 1
    else:
        world_size = jax.device_count()
        
    mesh = distribution.DeviceMesh(shape=(world_size,), axis_names=("model",))
    layout_map = distribution.LayoutMap(mesh)
    layout_map[".*token_embedding/embeddings"] = (None, "model")
    layout_map[".*query/kernel"] = ("model", None, None)
    layout_map[".*ffn_inner/kernel"] = (None, "model")
    layout_map[".*ffn_outer/kernel"] = ("model", None)
    
    model_parallel = distribution.ModelParallel(layout_map=layout_map)
    
    with model_parallel.scope():
        backbone = OPTBackbone(
            vocabulary_size=50272,
            num_layers=1,
            num_heads=4,
            hidden_dim=256,
            intermediate_dim=512,
            max_sequence_length=128,
            dropout=0.0,
        )
        model = OPTCausalLM(backbone=backbone)
        # Use Adam with matching epsilon and lower learning rate
        optimizer = keras.optimizers.Adam(learning_rate=1e-4, epsilon=1e-7)
        model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy")

    # --- WEIGHT SYNC LOGIC ---
    # We want both backends to start with the SAME weights.
    if backend == "jax":
        for v in model.weights:
            np.save(f"initial_weight_{v.path.replace('/', '_')}.npy", np.array(v.value))
    else:
        print(f"[torch Rank {rank}] Syncing weights from JAX source...")
        for v in model.weights:
            path = f"initial_weight_{v.path.replace('/', '_')}.npy"
            if os.path.exists(path):
                val = np.load(path)
                v.assign(val)

    # Helper to capture weight slice
    def get_weight_slice(model, pattern):
        for v in model.weights:
            if pattern in v.path:
                val = v.value
                if hasattr(val, "to_local"):
                    return val.to_local().detach().cpu().numpy()
                elif hasattr(val, "data") and hasattr(val.data, "to_local"):
                    return val.data.to_local().detach().cpu().numpy()
                elif hasattr(val, "detach"):
                    return val.detach().cpu().numpy()
                else:
                    return np.array(val)
        return None

    # Capture INITIAL weights (Step 0)
    weight_pattern = "query/kernel"
    init_weight = get_weight_slice(model, weight_pattern)
    if init_weight is not None:
        if backend == "torch":
            np.save(f"torch_weights_step0_rank{rank}.npy", init_weight)
        else:
             for i in range(jax.local_device_count()):
                v_target = [v for v in model.weights if weight_pattern in v.path][0]
                local_chunk = np.array(v_target.value.addressable_data(i))
                np.save(f"jax_weights_step0_rank{i}.npy", local_chunk)

    # 2. Load fixed data
    import tensorflow as tf
    token_ids = np.load("token_ids.npy")
    padding_mask = np.load("padding_mask.npy")
    targets = np.load("targets.npy")
    
    ds = tf.data.Dataset.from_tensors(({
        "token_ids": token_ids,
        "padding_mask": padding_mask,
    }, targets))
    
    dist_ds = model_parallel.distribute_dataset(ds)

    print(f"[{backend} Rank {rank}] Running 2 training steps...")
    batch = None
    for x, y in dist_ds.take(1):
        batch = (x, y)

    for step in [1, 2]:
        print(f"[{backend} Rank {rank}] Training Step {step}...")
        model.train_on_batch(batch[0], batch[1])
        
        target_weight = get_weight_slice(model, weight_pattern)
        if target_weight is not None:
            if backend == "torch":
                filename = f"torch_weights_step{step}_rank{rank}.npy"
                np.save(filename, target_weight)
            else:
                for i in range(jax.local_device_count()):
                    v_target = [v for v in model.weights if weight_pattern in v.path][0]
                    local_chunk = np.array(v_target.value.addressable_data(i))
                    filename = f"jax_weights_step{step}_rank{i}.npy"
                    np.save(filename, local_chunk)
        
            print(f"[{backend}] Step {step} Weight mean (rank/device 0): {np.mean(target_weight):.6f}")

    if backend == "torch":
        dist.destroy_process_group()

if __name__ == "__main__":
    run_test()
