import os
import sys
import time
import numpy as np

# Force CPU for everything - COMMENTED OUT TO ALLOW GPU USAGE
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

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
    # Convert to numpy to avoid device issues
    if hasattr(tokens, "cpu"):
        tokens = tokens.cpu().numpy()
    elif hasattr(tokens, "numpy"):
        tokens = tokens.numpy()
    
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
    
    # Use auto-detected devices (will pick up GPUs if available)
    devices = list_devices()
    print(f"INFO: Detected {len(devices)} devices: {devices}")
    
    if len(devices) < world_size:
        print(f"⚠️  Requested world_size {world_size} but only {len(devices)} devices found. Adjusting...")
        world_size = len(devices)

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
    
    sharded_model.fit(dataset, epochs=1, steps_per_epoch=1)
    print("JAX run completed.")

def _run_torch(rank, world_size):
    # Set Rank and World Size for Torch Distributed
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29505"
    
    # Prevent JAX from hogging GPUs during Torch runs
    os.environ["JAX_PLATFORMS"] = "cpu"
    
    import torch
    import torch.distributed as dist
    import keras
    import gc
    
    from keras.src.distribution.distribution_lib import AutoTPDistribution, DeviceMesh, initialize, list_devices
    
    print(f"[Process {rank}] Initializing distribution with world_size {world_size}")
    initialize(num_processes=world_size, process_id=rank)
    
    vocab_size = 10000
    dataset = get_dataset(vocab_size)
    
    # Create model on CPU to avoid GPU memory bottleneck before sharding
    print(f"[Process {rank}] Creating initial model on CPU...")
    with keras.device("cpu"):
        model = get_model(vocab_size)
    
    # Use real devices if available, otherwise fallback to CPU
    available_devices = list_devices()
    if not available_devices:
        available_devices = [f"cpu:{i}" for i in range(world_size)]
    
    devices = available_devices[:world_size]
    print(f"[Process {rank}] Using devices for mesh: {devices}")
    
    device_mesh = DeviceMesh(
        shape=(1, world_size), axis_names=("data", "model"), devices=devices
    )
    # Shard the model onto GPUs
    distribution = AutoTPDistribution(model, device_mesh=device_mesh)
    
    sharded_model = distribution.model
    sharded_model.compile(
        optimizer=keras.optimizers.Adam(2e-5),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    
    # Cleanup original model memory
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"[Process {rank}] Starting fit")
    sharded_model.fit(dataset, epochs=1, steps_per_epoch=1)
    print(f"[Process {rank}] Torch run completed.")
    
    if dist.is_initialized():
        dist.destroy_process_group()

def run_backend(backend, world_size=2):
    os.environ["KERAS_BACKEND"] = backend
    
    import keras
    from keras.src.distribution.distribution_lib import list_devices
    
    # Use provided world_size or default to 2
    if world_size is None:
        world_size = 2
    
    print(f"\n{'='*20} BACKEND: {backend} (World Size: {world_size}) {'='*20}")
    
    if backend == "jax":
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
        _run_jax(world_size)
    elif backend == "torch":
        import torch
        torch.multiprocessing.spawn(_run_torch, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="torch", choices=["torch", "jax"])
    parser.add_argument("--world_size", type=int, default=None)
    args = parser.parse_args()
    
    run_backend(args.backend, args.world_size)
