import os
import sys

# Ensure we use the local keras source
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
)

import argparse
import hashlib

import numpy as np
import torch

# Configuration
BATCH_SIZE = 8
SEQ_LENGTH = 32
VOCAB_SIZE = 50272  # OPT default
NUM_DEVICES = 2
SEED = 1337


def create_model():
    import keras_hub

    import keras

    keras.utils.set_random_seed(SEED)
    model = keras_hub.models.OPTCausalLM.from_preset(
        "opt_125m_en",
        load_weights=False,
    )
    initialize_model_weights(model)
    # Use random init for speed
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=2e-5),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    )
    return model


def initialize_model_weights(model):
    for weight in model.weights:
        shape = tuple(weight.shape)
        if len(shape) == 1:
            values = np.zeros(shape, dtype="float32")
            if "scale" in weight.path or "gamma" in weight.path:
                values.fill(1.0)
        else:
            path_seed = int.from_bytes(
                hashlib.sha256(weight.path.encode("utf-8")).digest()[:8],
                "little",
            )
            rng = np.random.default_rng(SEED + path_seed)
            values = rng.normal(0.0, 0.02, size=shape).astype("float32")
        weight.assign(values)


def create_batch():
    rng = np.random.default_rng(SEED)
    token_ids = rng.integers(
        0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LENGTH), dtype=np.int32
    )
    padding_mask = np.ones((BATCH_SIZE, SEQ_LENGTH), dtype="int32")
    y = rng.integers(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LENGTH), dtype=np.int32)
    return {"token_ids": token_ids, "padding_mask": padding_mask}, y


def run_training_torch(rank, world_size):
    # Each process needs its own backend setup
    os.environ["KERAS_BACKEND"] = "torch"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    # Re-import inside process
    import keras
    from keras.src.backend.torch.distribution_lib import initialize
    from keras.src.distribution.distribution_lib import AutoTPDistribution
    from keras.src.distribution.distribution_lib import DeviceMesh

    initialize()

    devices = keras.distribution.list_devices()
    if rank == 0:
        print(f"Devices found: {devices}")

    mesh = DeviceMesh(
        shape=(1, world_size), axis_names=("data", "model"), devices=devices
    )

    template_model = create_model()
    dist_strategy = AutoTPDistribution(template_model, device_mesh=mesh)
    model = dist_strategy.model

    # Re-compile sharded model
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=2e-5),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    )

    x, y = create_batch()

    print(f"[Rank {rank}] Training one step...")
    loss = model.train_on_batch(x, y)
    print(f"✅ [Rank {rank}] Torch Step Complete. Loss: {loss}")


def run_jax():
    os.environ["KERAS_BACKEND"] = "jax"
    # Simulate multi-device if on CPU
    os.environ["XLA_FLAGS"] = (
        f"--xla_force_host_platform_device_count={NUM_DEVICES}"
    )

    import keras
    from keras.src.distribution.distribution_lib import AutoTPDistribution
    from keras.src.distribution.distribution_lib import DeviceMesh
    from keras.src.distribution.distribution_lib import list_devices

    devices = list_devices()
    print(f"Devices found: {devices}")
    mesh = DeviceMesh(
        shape=(1, len(devices)), axis_names=("data", "model"), devices=devices
    )

    template_model = create_model()
    dist_strategy = AutoTPDistribution(template_model, device_mesh=mesh)
    model = dist_strategy.model

    # Re-compile sharded model
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=2e-5),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    )

    x, y = create_batch()

    print("Training one step...")
    loss = model.train_on_batch(x, y)
    print(f"✅ JAX Step Complete. Loss: {loss}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("backend", choices=["jax", "torch"], default="torch")
    args = parser.parse_args()

    if args.backend == "torch":
        torch.multiprocessing.spawn(
            run_training_torch,
            args=(NUM_DEVICES,),
            nprocs=NUM_DEVICES,
            join=True,
        )
    else:
        run_jax()
