"""A Keras on JAX training script to learn sin(x) with distributed training."""

from collections.abc import Sequence
import os
import typing

# Set Keras backend to JAX before importing Keras
os.environ["KERAS_BACKEND"] = "jax"

from absl import app
from absl import flags
from absl import logging
import jax
import keras
import numpy as np
from tensorflow import data as tf_data


FLAGS = flags.FLAGS
BATCH_SIZE = flags.DEFINE_integer("batch_size", 64, "Batch size for training.")
EPOCHS = flags.DEFINE_integer("epochs", 50, "Number of training epochs.")
LEARNING_RATE = flags.DEFINE_float(
    "learning_rate", 1e-3, "Optimizer learning rate."
)
VALIDATION_SPLIT = flags.DEFINE_float(
    "validation_split",
    0.2,
    "Fraction of training data for validation.",
)
EARLY_STOPPING_PATIENCE = flags.DEFINE_integer(
    "early_stopping_patience",
    10,
    "Patience for early stopping.",
)
MODEL_CHECKPOINT_PATH = flags.DEFINE_string(
    "model_checkpoint_path",
    "best_model_sin_distributed.keras",
    "Path to save model checkpoints.",
)

NUM_FEATURES: typing.Final[int] = 1  # Input is a single value 'x'
NUM_OUTPUTS: typing.Final[int] = 1  # Output is a single value 'sin(x)'
INPUT_SHAPE: typing.Final[tuple[int, ...]] = (
    NUM_FEATURES,
)  # Input shape for a dense layer
EXPECTED_MINIMUM_DEVICES: typing.Final[int] = 4


def load_and_preprocess_data() -> tuple[
    tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]
]:
    """Generates and returns a synthetic dataset for y = sin(x)."""
    logging.info("Generating synthetic dataset for y = sin(x)...")
    num_samples_total = 2000

    # Generate x values in a range, e.g., -2*pi to 2*pi
    x_values = np.linspace(-2 * np.pi, 2 * np.pi, num_samples_total).astype(
        "float32"
    )

    # Calculate y = sin(x)
    y_values = np.sin(x_values).astype("float32")

    # Add some Gaussian noise to the target variable to make learning more
    # realistic
    noise_level = 0.05
    rng = np.random.default_rng(seed=42)
    y_with_noise = y_values + noise_level * rng.standard_normal(
        num_samples_total, dtype=np.float32
    )

    # Reshape x_values to be (num_samples, 1) as Keras dense layers expect input
    # shape (batch_size, features)
    x_values = x_values.reshape(-1, NUM_FEATURES)

    # Shuffle the data before splitting.
    indices = np.arange(num_samples_total)
    rng.shuffle(indices)
    shuffled_x, shuffled_y = x_values[indices], y_with_noise[indices]

    # Split the data into training and test sets.
    test_set_fraction = 0.2
    num_test_samples = int(num_samples_total * test_set_fraction)
    num_train_samples = num_samples_total - num_test_samples

    x_train, y_train = (
        shuffled_x[:num_train_samples],
        shuffled_y[:num_train_samples],
    )
    x_test, y_test = (
        shuffled_x[num_train_samples:],
        shuffled_y[num_train_samples:],
    )

    logging.info("Total samples: %s", num_samples_total)
    logging.info(
        "x_train shape: %s, y_train shape: %s", x_train.shape, y_train.shape
    )
    logging.info(
        "x_test shape: %s, y_test shape: %s", x_test.shape, y_test.shape
    )

    return (x_train, y_train), (x_test, y_test)


def create_model(input_shape: tuple[int, ...], num_outputs: int) -> keras.Model:
    """Creates and returns a Keras Dense model for sin(x) regression."""
    logging.info("Creating regression model for sin(x)...")
    return keras.Sequential(
        [
            keras.layers.Input(shape=input_shape, name="input_layer"),
            keras.layers.Dense(64, activation="relu", name="dense_hidden_1"),
            keras.layers.Dense(64, activation="relu", name="dense_hidden_2"),
            keras.layers.Dense(
                num_outputs, activation="linear", name="output_dense"
            ),
        ]
    )


def train_model(
    model: keras.Model,
    *,
    train_dataset: tf_data.Dataset,
    validation_dataset: tf_data.Dataset,
    epochs: int,
    model_checkpoint_path: str,
    early_stopping_patience: int,
) -> keras.callbacks.History:
    """Trains the compiled Keras model using tf.data.Dataset."""
    logging.info("Setting up callbacks...")
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=model_checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=early_stopping_patience, verbose=1
        ),
        keras.callbacks.TensorBoard(log_dir="./logs_sin_distributed"),
    ]

    logging.info("Starting training for %s epochs...", epochs)
    return model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=validation_dataset,
        callbacks=callbacks,
        verbose=1,
    )


def evaluate_model(
    model: keras.Model, test_dataset: tf_data.Dataset
) -> list[float]:
    """Evaluates the trained model on the test dataset."""
    logging.info("Evaluating model on the test set...")
    score = model.evaluate(test_dataset, verbose=0)
    logging.info("Test Loss (MSE): %.4f", score[0])
    logging.info("Test Mean Absolute Error (MAE): %.4f", score[1])
    return score


def main(argv: Sequence[str]) -> None:
    """Main training and evaluation script for sin(x) approximation."""
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    logging.set_verbosity(logging.INFO)

    devices = jax.local_devices()
    num_devices = len(devices)
    logging.info("Running on %s devices: %s", num_devices, devices)

    if num_devices < EXPECTED_MINIMUM_DEVICES:
        logging.warning(
            "Expected at least %d devices, but found %d. Proceeding anyway for demo purposes."
            % (EXPECTED_MINIMUM_DEVICES, num_devices)
        )

    # 1. Create a 1D DeviceMesh for data parallelism.
    mesh_1d = keras.distribution.DeviceMesh(
        shape=(num_devices,), axis_names=("data",), devices=devices
    )

    # 2. As per the docs, DataParallel is initialized with the mesh directly.
    # The `TensorLayout` is handled implicitly for data parallelism.
    distribution = keras.distribution.DataParallel(device_mesh=mesh_1d)
    keras.distribution.set_distribution(distribution)

    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()

    # 3. Convert NumPy arrays to tf.data.Dataset. This is the standard and
    # most robust way to handle input pipelines for distributed training.
    global_batch_size = BATCH_SIZE.value * num_devices
    logging.info("Global batch size across all devices: %d", global_batch_size)

    # Manually split training data for validation
    num_train_samples = len(x_train)
    num_val_samples = int(num_train_samples * VALIDATION_SPLIT.value)

    train_dataset = tf_data.Dataset.from_tensor_slices((x_train, y_train))
    val_dataset = train_dataset.take(num_val_samples)
    train_dataset = train_dataset.skip(num_val_samples)
    test_dataset = tf_data.Dataset.from_tensor_slices((x_test, y_test))

    # Shuffle, batch, and prefetch for performance
    train_dataset = (
        train_dataset.shuffle(buffer_size=num_train_samples)
        .batch(global_batch_size)
        .prefetch(tf_data.AUTOTUNE)
    )
    val_dataset = val_dataset.batch(global_batch_size).prefetch(
        tf_data.AUTOTUNE
    )
    test_dataset = test_dataset.batch(global_batch_size).prefetch(
        tf_data.AUTOTUNE
    )

    # 4. Create and compile the model within the distribution scope.
    with distribution.scope():
        model = create_model(input_shape=INPUT_SHAPE, num_outputs=NUM_OUTPUTS)
        model.summary(print_fn=logging.info)
        logging.info("Compiling regression model...")
        model.compile(
            loss=keras.losses.MeanSquaredError(),
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE.value),
            metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
        )

    logging.info("Training model to approximate sin(x)...")

    # 5. Train the model using the prepared datasets.
    train_model(
        model,
        train_dataset=train_dataset,
        validation_dataset=val_dataset,
        epochs=EPOCHS.value,
        model_checkpoint_path=MODEL_CHECKPOINT_PATH.value,
        early_stopping_patience=EARLY_STOPPING_PATIENCE.value,
    )
    logging.info("Model training complete.")

    # 6. Load and evaluate the final model.
    with distribution.scope():
        logging.info(
            "Loading the best model from: %s", MODEL_CHECKPOINT_PATH.value
        )
        model = keras.models.load_model(MODEL_CHECKPOINT_PATH.value)

    evaluate_model(model, test_dataset=test_dataset)
    logging.info("Script finished successfully.")


if __name__ == "__main__":
    app.run(main)
