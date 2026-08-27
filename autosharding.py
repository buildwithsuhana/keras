import os
import sys

# Set fixed environment variables before any other imports
os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
os.environ["JAX_PLATFORMS"] = "cpu"

import argparse
import time

import numpy as np
import keras

# Set fixed seed for reproducibility
keras.utils.set_random_seed(42)

# Ensure we use the local keras source
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
)


def get_model(vocab_size):
    import keras_nlp
    import keras
    
    keras.backend.set_floatx("float32")

    opt_model = keras_nlp.models.OPTCausalLM.from_preset(
        "opt_125m_en",
        load_weights=False,
        preprocessor=None,
        dropout=0.0,
    )
    opt_model.backbone.token_embedding.vocabulary_size = vocab_size
    opt_model.backbone.token_embedding.embeddings_initializer = (
        keras.initializers.RandomNormal(stddev=0.02)
    )
    dummy_input = {
        "token_ids": np.ones((1, 1), dtype="int32"),
        "padding_mask": np.ones((1, 1), dtype="int32"),
    }
    opt_model.backbone(dummy_input)

    opt_model.compile(
        optimizer=keras.optimizers.SGD(1e-5),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    )
    return opt_model


def get_dataset(vocab_size):
    import keras_nlp
    import tensorflow as tf
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
    import keras
    from keras import ops
    from keras.src.distribution.distribution_lib import AutoTPDistribution
    from keras.src.distribution.distribution_lib import DeviceMesh
    from keras.src.distribution.distribution_lib import list_devices

    vocab_size = 10000
    dataset = get_dataset(vocab_size)
    first_batch = next(iter(dataset))
    x, y = first_batch
    x = {k: np.array(v) for k, v in x.items()}
    y = np.array(y)
    
    model = get_model(vocab_size)

    weights_file = f"shared_weights_{vocab_size}.weights.h5"
    if not os.path.exists(weights_file):
        _ = model(x)
        model.save_weights(weights_file)
    model.load_weights(weights_file)
    
    # Verify weights
    first_weight = model.weights[0]
    print(f"DEBUG: JAX first weight (10 elements): {np.array(first_weight).flatten()[:10]}")

    devices = list_devices("cpu")
    device_mesh = DeviceMesh(shape=(1, world_size), axis_names=("data", "model"), devices=devices[:world_size])
    distribution = AutoTPDistribution(model, device_mesh=device_mesh)
    sharded_model = distribution.model
    
    sharded_model.compile(optimizer=keras.optimizers.SGD(1e-5), loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))

    # Initial Loss
    logits = sharded_model(x)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    initial_loss = loss_fn(y, logits)
    print(f"INITIAL_LOSS: {float(initial_loss):.12f}")

    # Use train_on_batch for stable reporting
    fit_loss = sharded_model.train_on_batch(x, y)
    # Explicitly convert to numpy for reliable printing
    fit_loss_val = float(np.array(fit_loss))
    print(f"FIT_LOSS: {fit_loss_val:.12f}")
    
    logits_after = sharded_model(x)
    final_loss = loss_fn(y, logits_after)
    print(f"FINAL_LOSS_AFTER_FIT: {float(final_loss):.12f}")


def find_free_port():
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _run_torch(rank, world_size, port):
    import os
    import sys
    
    # Use the local keras source
    sys.path.insert(
        0, os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
    )
    
    # Force CPU for all operations to avoid MPS issues with collectives
    os.environ["KERAS_BACKEND"] = "torch"
    os.environ["KERAS_TORCH_DEVICE"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # Ensure JAX doesn't try to initialize anything
    os.environ["JAX_PLATFORMS"] = "cpu"
    
    import torch
    # Ensure torch doesn't try to use MPS
    if hasattr(torch.backends, "mps"):
         torch.backends.mps.is_available = lambda: False
    
    if hasattr(torch, "set_default_device"):
        torch.set_default_device("cpu")

    import keras
    keras.config.set_backend("torch")
    
    import torch.distributed as dist
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    from keras.src.distribution.distribution_lib import AutoTPDistribution, DeviceMesh, initialize, list_devices

    initialize()
    vocab_size = 10000
    dataset = get_dataset(vocab_size)
    first_batch = next(iter(dataset))
    x, y = first_batch
    x = {k: np.array(v) for k, v in x.items()}
    y = np.array(y)
    
    model = get_model(vocab_size)

    weights_file = f"shared_weights_{vocab_size}.weights.h5"
    while not os.path.exists(weights_file):
        if rank == 0:
             _ = model(x)
             model.save_weights(weights_file)
        else:
             time.sleep(1)
    model.load_weights(weights_file)
    
    # Verify weights
    if rank == 0:
        first_weight = model.weights[0]
        print(f"DEBUG: TORCH first weight (10 elements): {np.array(first_weight).flatten()[:10]}")

    devices = [f"cpu:{i}" for i in range(world_size)]
    device_mesh = DeviceMesh(shape=(1, world_size), axis_names=("data", "model"), devices=devices[:world_size])
    distribution = AutoTPDistribution(model, device_mesh=device_mesh)
    sharded_model = distribution.model
    sharded_model.compile(optimizer=keras.optimizers.SGD(1e-5), loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))

    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    # Initial
    import torch
    with torch.no_grad():
        logits = sharded_model(x)
        initial_loss = loss_fn(y, logits)
    
    # Use standard train_on_batch
    fit_loss = sharded_model.train_on_batch(x, y)
    
    with torch.no_grad():
        logits_after = sharded_model(x)
        final_loss = loss_fn(y, logits_after)

    if rank == 0:
        from keras import ops
        print(f"INITIAL_LOSS: {float(ops.convert_to_numpy(initial_loss)):.12f}")
        fit_loss_val = float(ops.convert_to_numpy(fit_loss))
        print(f"FIT_LOSS: {fit_loss_val:.12f}")
        print(f"FINAL_LOSS_AFTER_FIT: {float(ops.convert_to_numpy(final_loss)):.12f}")

    if dist.is_initialized():
        dist.destroy_process_group()


def run_backend(backend, world_size=2):
    os.environ["KERAS_BACKEND"] = backend
    if backend == "jax":
        os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={world_size}"
        os.environ["JAX_PLATFORMS"] = "cpu"
        _run_jax(world_size)
    elif backend == "torch":
        import torch
        port = find_free_port()
        torch.multiprocessing.spawn(_run_torch, args=(world_size, port), nprocs=world_size, join=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="torch", choices=["torch", "jax"])
    args = parser.parse_args()
    run_backend(args.backend)
