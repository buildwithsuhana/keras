import os
import time

# --- Force CPU and Simulate 2 Devices ---
# This block MUST come before any tensorflow or keras imports.
# 1. Hide GPUs from TensorFlow/JAX.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# 2. Tell XLA to create 2 virtual CPU devices.
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

import keras_nlp
import matplotlib.pyplot as plt
import tensorflow as tf

import keras

# --- Import your custom distribution strategy ---
from keras.src.distribution.distribution_lib import AutoTPDistribution
from keras.src.distribution.distribution_lib import DeviceMesh
from keras.src.distribution.distribution_lib import list_devices

# --- Configuration ---
BATCH_SIZE = 8
SEQ_LENGTH = 128
EPOCHS = 2
VOCAB_SIZE = 10000

# --- 1. Data Loading and Preprocessing ---
print("INFO: Loading Tiny Shakespeare dataset...")
path = keras.utils.get_file(
    "tiny_shakespeare.txt",
    origin="https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt",
)
with open(path) as f:
    text_data = f.read()

print("INFO: Building vocabulary...")
tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    vocabulary_size=VOCAB_SIZE,
    lowercase=True,
)
tokenizer.train_on_batch([text_data])

print("INFO: Preprocessing data into sequences...")
tokens = tokenizer.tokenize(text_data)
dataset = tf.data.Dataset.from_tensor_slices(tokens)
dataset = dataset.batch(SEQ_LENGTH + 1, drop_remainder=True)


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


train_dataset = dataset.map(split_input_target).batch(
    BATCH_SIZE, drop_remainder=True
)


# --- 2. Model Definition ---
def create_model():
    """Creates a new instance of the OPT-125M model."""
    print("INFO: Creating OPT-125M model...")
    opt_model = keras_nlp.models.OPTCausalLM.from_preset(
        "opt_125m_en",
        load_weights=False,
    )
    opt_model.backbone.token_embedding.embeddings_initializer = (
        keras.initializers.RandomNormal(stddev=0.02)
    )
    opt_model.backbone.token_embedding._built = False
    opt_model.backbone.token_embedding.vocabulary_size = VOCAB_SIZE
    opt_model.preprocessor.tokenizer.vocabulary_size = VOCAB_SIZE

    opt_model.compile(
        optimizer=keras.optimizers.Adam(2e-5),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    return opt_model


# --- 3. Scenario A: Distributed Training with AutoTPDistribution on Virtual CPUs ---
print("\n" + "=" * 50)
print(" S T A R T I N G   D I S T R I B U T E D   T R A I N I N G ")
print("=" * 50)

# Keras will now automatically detect the 2 virtual CPU devices from XLA_FLAGS
DEVICES = list_devices()
print(f"INFO: Detected {len(DEVICES)} virtual CPU devices: {DEVICES}")

# Create the DeviceMesh and the distribution strategy
device_mesh = DeviceMesh(
    shape=(len(DEVICES),), axis_names=("model",), devices=DEVICES
)
distribution = AutoTPDistribution(device_mesh=device_mesh)

model_to_shard = create_model()
sharded_model = distribution.shard(model_to_shard)

start_time = time.time()
distributed_history = sharded_model.fit(
    train_dataset,
    epochs=EPOCHS,
)
distributed_time = time.time() - start_time
print(f"Distributed training took: {distributed_time:.2f} seconds.")


# --- 4. Scenario B: Single Device Training on one CPU ---
print("\n" + "=" * 50)
print(" S T A R T I N G   S I N G L E   D E V I C E   T R A I N I N G ")
print("=" * 50)

single_device_model = create_model()

start_time = time.time()
single_device_history = single_device_model.fit(
    train_dataset,
    epochs=EPOCHS,
)
single_device_time = time.time() - start_time
print(f"Single device training took: {single_device_time:.2f} seconds.")


# --- 5. Comparison and Plotting ---
print("\n" + "=" * 50)
print(" C O M P A R I N G   R E S U L T S ")
print("=" * 50)

plt.figure(figsize=(10, 6))
plt.plot(
    distributed_history.history["loss"],
    label="Distributed Training Loss (2 Virtual CPUs)",
    marker="o",
)
plt.plot(
    single_device_history.history["loss"],
    label="Single Device Loss (Baseline)",
    marker="x",
    linestyle="--",
)
plt.title("Training Loss Comparison: OPT-125M on Tiny Shakespeare")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.xticks(range(EPOCHS))
plt.legend()
plt.grid(True)
plt.show()

final_dist_loss = distributed_history.history["loss"][-1]
final_single_loss = single_device_history.history["loss"][-1]

print(f"\nFinal Distributed Loss: {final_dist_loss:.4f}")
print(f"Final Single Device Loss: {final_single_loss:.4f}")

if abs(final_dist_loss - final_single_loss) < 1e-3:
    print(
        "\n✅ The final loss values are nearly identical. The sharding implementation is correct!"
    )
else:
    print(
        "\n⚠️ The final loss values are different. There might be a bug in the sharding logic."
    )
