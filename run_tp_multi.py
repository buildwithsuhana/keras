import os
import sys
import time
import numpy as np

# Ensure we use the local keras source
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

def get_model(vocab_size):
    import keras
    import keras_nlp
    print("INFO: Creating OPT-125M model...")
    opt_model = keras_nlp.models.OPTCausalLM.from_preset(
        "opt_125m_en",
        load_weights=False,
        preprocessor=None,
    )
    opt_model.backbone.token_embedding.embeddings_initializer = (
        keras.initializers.RandomNormal(stddev=0.02)
    )
    opt_model.backbone.token_embedding._built = False
    opt_model.backbone.token_embedding.vocabulary_size = vocab_size

    opt_model.compile(
        optimizer=keras.optimizers.Adam(2e-5),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    return opt_model

def get_dataset(vocab_size):
    import tensorflow as tf
    import keras_nlp
    import keras
    
    path = keras.utils.get_file(
        "tiny_shakespeare.txt",
        origin="https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt",
    )
    with open(path) as f:
        text_data = f.read()

    vocab = keras_nlp.tokenizers.compute_word_piece_vocabulary(
        [path],
        vocabulary_size=vocab_size,
        lowercase=True,
    )
    tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
        vocabulary=vocab,
        lowercase=True,
    )

    tokens = tokenizer.tokenize(text_data)
    tokens = keras.ops.convert_to_numpy(tokens)
    dataset = tf.data.Dataset.from_tensor_slices(tokens)
    dataset = dataset.batch(128 + 1, drop_remainder=True)

    def split_input_target(chunk):
        input_text = chunk[:-1]
        target_text = chunk[1:]
        return {
            "token_ids": input_text,
            "padding_mask": tf.ones_like(input_text),
        }, target_text

    return dataset.map(split_input_target).batch(8, drop_remainder=True)

def _run_jax(world_size):
    print(f"\n--- Running JAX with {world_size} devices ---")
    import keras
    from keras.src.distribution.distribution_lib import AutoTPDistribution, DeviceMesh, list_devices
    
    vocab_size = 10000
    dataset = get_dataset(vocab_size)
    model = get_model(vocab_size)
    
    devices = list_devices("cpu")
    print(f"INFO: Detected {len(devices)} devices: {devices}")
    
    device_mesh = DeviceMesh(
        shape=(1, world_size), axis_names=("data", "model"), devices=devices[:world_size]
    )
    distribution = AutoTPDistribution(model, device_mesh=device_mesh)
    
    sharded_model = distribution.model
    sharded_model.compile(
        optimizer=keras.optimizers.Adam(2e-5),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    
    print("\n--- Model Summary with Sharding (JAX) ---")
    sharded_model.summary(show_sharding=True)
    
    sharded_model.fit(dataset, epochs=1, steps_per_epoch=1)
    print("JAX run completed.")

def _run_torch(rank, world_size):
    # This will be called by mp.spawn
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29505"
    
    # Force Keras to use the correct local device for this process
    import torch
    if torch.cuda.is_available():
        os.environ["KERAS_TORCH_DEVICE"] = f"cuda:{rank}"
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        os.environ["KERAS_TORCH_DEVICE"] = f"xpu:{rank}"
    
    import torch.distributed as dist
    import keras
    from keras.src.distribution.distribution_lib import AutoTPDistribution, DeviceMesh, list_devices, initialize
    
    print(f"[Process {rank}] Initializing distribution")
    initialize()
    
    vocab_size = 10000
    # In multi-process, we might want to avoid all processes downloading/processing data
    # but for this example we'll just do it.
    dataset = get_dataset(vocab_size)
    model = get_model(vocab_size)
    
    if torch.cuda.is_available():
        devices = list_devices("gpu")
    else:
        devices = list_devices("cpu")
        # Mock devices if not enough
        if len(devices) < world_size:
            devices = [f"cpu:{i}" for i in range(world_size)]

    print(f"[Process {rank}] Using devices: {devices[:world_size]}")
    
    device_mesh = DeviceMesh(
        shape=(1, world_size), axis_names=("data", "model"), devices=devices[:world_size]
    )
    distribution = AutoTPDistribution(model, device_mesh=device_mesh)
    
    sharded_model = distribution.model
    sharded_model.compile(
        optimizer=keras.optimizers.Adam(2e-5),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

    print(f"\n[Process {rank}] --- Model Summary with Sharding (Torch) ---")
    if rank == 0:
        sharded_model.summary(show_sharding=True)

    # Explicit build to ensure variables are registered with optimizer
    print(f"[Process {rank}] Explicitly building sharded model...")
    import tensorflow as tf
    dummy_input = {
        "token_ids": np.ones((8, 128), dtype="int32"),
        "padding_mask": np.ones((8, 128), dtype="int32"),
    }
    sharded_model(dummy_input)
    
    print(f"[Process {rank}] Starting fit")
    sharded_model.fit(dataset, epochs=1, steps_per_epoch=1)
    print(f"[Process {rank}] Torch run completed.")
    
    if dist.is_initialized():
        dist.destroy_process_group()

def run_backend(backend, world_size=2):
    print(f"{'='*20} BACKEND: {backend} {'='*20}")
    os.environ["KERAS_BACKEND"] = backend
    if backend == "jax":
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
        # Force CPU simulation for JAX if not enough GPUs
        os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={world_size}"
        os.environ["JAX_PLATFORMS"] = "cpu"
        _run_jax(world_size)
    elif backend == "torch":
        import torch
        # Note: torch.multiprocessing.spawn handles the rank argument automatically
        torch.multiprocessing.spawn(_run_torch, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="torch", choices=["torch", "jax"])
    args = parser.parse_args()
    
    run_backend(args.backend)
